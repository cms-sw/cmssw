/*----------------------------------------------------------------------

Test program for edm::Event.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"

#include <cppunit/extensions/HelperMacros.h>

#include "boost/shared_ptr.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

using namespace edm;

// This is a gross hack, to allow us to test the event
namespace edm {
  class EDProducer {
  public:
    static void commitEvent(Event& e) { e.commit_(); }

  };
}


class testEvent: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testEvent);
  CPPUNIT_TEST(emptyEvent);
  CPPUNIT_TEST(getByLabelFromEmpty);
  CPPUNIT_TEST(putAnIntProduct);
  CPPUNIT_TEST(putAndGetAnIntProduct);
  CPPUNIT_TEST(getByProductID);
  CPPUNIT_TEST(transaction);
  CPPUNIT_TEST(getByLabel);
  CPPUNIT_TEST(getManyByType);
  CPPUNIT_TEST(printHistory);
  CPPUNIT_TEST(deleteProduct);
  CPPUNIT_TEST_SUITE_END();

 public:
  testEvent();
  ~testEvent();
  void setUp();
  void tearDown();
  void emptyEvent();
  void getByLabelFromEmpty();
  void putAnIntProduct();
  void putAndGetAnIntProduct();
  void getByProductID();
  void transaction();
  void getByLabel();
  void getManyByType();
  void printHistory();
  void deleteProduct();

 private:

  template <class T>
  void registerProduct(std::string const& tag,
                       std::string const& moduleLabel,
                       std::string const& moduleClassName,
                       std::string const& processName,
                       std::string const& productInstanceName);

  template <class T>
  void registerProduct(std::string const& tag,
                       std::string const& moduleLabel,
                       std::string const& moduleClassName,
                       std::string const& processName) {
    std::string productInstanceName;
    registerProduct<T>(tag, moduleLabel, moduleClassName, processName,
                       productInstanceName);
  }

  template <class T>
  ProductID addProduct(std::auto_ptr<T> product,
                       std::string const& tag,
                       std::string const& productLabel = std::string());

  boost::shared_ptr<ProductRegistry>   availableProducts_;
  boost::shared_ptr<BranchIDListHelper> branchIDListHelper_;
  boost::shared_ptr<EventPrincipal>    principal_;
  boost::shared_ptr<Event>             currentEvent_;
  boost::shared_ptr<ModuleDescription> currentModuleDescription_;
  typedef std::map<std::string, ModuleDescription> modCache_t;
  typedef modCache_t::iterator iterator_t;

  modCache_t moduleDescriptions_;
  std::vector<boost::shared_ptr<ProcessConfiguration> > processConfigurations_;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEvent);

EventID   make_id() { return EventID(2112, 1, 25); }
Timestamp make_timestamp() { return Timestamp(1); }

template <class T>
void
testEvent::registerProduct(std::string const& tag,
                           std::string const& moduleLabel,
                            std::string const& moduleClassName,
                            std::string const& processName,
                           std::string const& productInstanceName) {
  if (!availableProducts_)
    availableProducts_.reset(new ProductRegistry());

  ParameterSet moduleParams;
  moduleParams.template addParameter<std::string>("@module_type", moduleClassName);
  moduleParams.template addParameter<std::string>("@module_label", moduleLabel);
  moduleParams.registerIt();

  ParameterSet processParams;
  processParams.template addParameter<std::string>("@process_name", processName);
  processParams.template addParameter<ParameterSet>(moduleLabel, moduleParams);
  processParams.registerIt();

  ProcessConfiguration process(processName, processParams.id(), getReleaseVersion(), getPassID());

  boost::shared_ptr<ProcessConfiguration> processX(new ProcessConfiguration(process));
  processConfigurations_.push_back(processX);

  TypeWithDict product_type(typeid(T));

  BranchDescription branch(InEvent,
                           moduleLabel,
                           processName,
                           product_type.userClassName(),
                           product_type.friendlyClassName(),
                           productInstanceName,
                           moduleClassName,
                           moduleParams.id(),
                           product_type
                        );

  moduleDescriptions_[tag] = ModuleDescription(moduleParams.id(), moduleClassName, moduleLabel, processX.get());
  availableProducts_->addProduct(branch);
}

// Add the given product, of type T, to the current event,
// as if it came from the module specified by the given tag.
template <class T>
ProductID
testEvent::addProduct(std::auto_ptr<T> product,
                      std::string const& tag,
                      std::string const& productLabel) {
  iterator_t description = moduleDescriptions_.find(tag);
  if (description == moduleDescriptions_.end())
    throw edm::Exception(errors::LogicError)
      << "Failed to find a module description for tag: "
      << tag << '\n';

  Event temporaryEvent(*principal_, description->second);
  OrphanHandle<T> h = temporaryEvent.put(product, productLabel);
  ProductID id = h.id();
  EDProducer::commitEvent(temporaryEvent);
  return id;
}

testEvent::testEvent() :
  availableProducts_(new ProductRegistry()),
  branchIDListHelper_(new BranchIDListHelper()),
  principal_(),
  currentEvent_(),
  currentModuleDescription_(),
  moduleDescriptions_(),
  processConfigurations_() {

  typedef edmtest::IntProduct prod_t;
  typedef std::vector<edmtest::Thing> vec_t;

  registerProduct<prod_t>("nolabel_tag",   "modOne",   "IntProducer",   "EARLY");
  registerProduct<prod_t>("int1_tag",      "modMulti", "IntProducer",   "EARLY", "int1");
  registerProduct<prod_t>("int1_tag_late", "modMulti", "IntProducer",   "LATE",  "int1");
  registerProduct<prod_t>("int2_tag",      "modMulti", "IntProducer",   "EARLY", "int2");
  registerProduct<prod_t>("int3_tag",      "modMulti", "IntProducer",   "EARLY");
  registerProduct<vec_t>("thing",          "modthing", "ThingProducer", "LATE");
  registerProduct<vec_t>("thing2",         "modthing", "ThingProducer", "LATE",  "inst2");

  // Fake up the production of a single IntProduct from an IntProducer
  // module, run in the 'CURRENT' process.
  ParameterSet moduleParams;
  std::string moduleLabel("modMulti");
  std::string moduleClassName("IntProducer");
  moduleParams.addParameter<std::string>("@module_type", moduleClassName);
  moduleParams.addParameter<std::string>("@module_label", moduleLabel);
  moduleParams.registerIt();

  ParameterSet processParams;
  std::string processName("CURRENT");
  processParams.addParameter<std::string>("@process_name", processName);
  processParams.addParameter(moduleLabel, moduleParams);
  processParams.registerIt();

  ProcessConfiguration process(processName, processParams.id(), getReleaseVersion(), getPassID());

  TypeWithDict product_type(typeid(prod_t));

  boost::shared_ptr<ProcessConfiguration> processX(new ProcessConfiguration(process));
  processConfigurations_.push_back(processX);
  currentModuleDescription_.reset(new ModuleDescription(moduleParams.id(), moduleClassName, moduleLabel, processX.get()));

  std::string productInstanceName("int1");

  BranchDescription branch(InEvent,
                           moduleLabel,
                           processName,
                           product_type.userClassName(),
                           product_type.friendlyClassName(),
                           productInstanceName,
                           moduleClassName,
                           moduleParams.id(),
                           product_type
                        );

  availableProducts_->addProduct(branch);

  // Freeze the product registry before we make the Event.
  availableProducts_->setFrozen();
  branchIDListHelper_->updateRegistries(*availableProducts_);
}

testEvent::~testEvent() {
}

void testEvent::setUp() {

  edm::RootAutoLibraryLoader::enable();
  // First build a fake process history, that says there
  // were previous processes named "EARLY" and "LATE".
  // This takes several lines of code but other than
  // the process names none of it is used or interesting.
  ParameterSet moduleParamsEarly;
  std::string moduleLabelEarly("currentModule");
  std::string moduleClassNameEarly("IntProducer");
  moduleParamsEarly.addParameter<std::string>("@module_type", moduleClassNameEarly);
  moduleParamsEarly.addParameter<std::string>("@module_label", moduleLabelEarly);
  moduleParamsEarly.registerIt();

  ParameterSet processParamsEarly;
  std::string processNameEarly("EARLY");
  processParamsEarly.addParameter<std::string>("@process_name", processNameEarly);
  processParamsEarly.addParameter(moduleLabelEarly, moduleParamsEarly);
  processParamsEarly.registerIt();

  ProcessConfiguration processEarly("EARLY", processParamsEarly.id(), getReleaseVersion(), getPassID());

  ParameterSet moduleParamsLate;
  std::string moduleLabelLate("currentModule");
  std::string moduleClassNameLate("IntProducer");
  moduleParamsLate.addParameter<std::string>("@module_type", moduleClassNameLate);
  moduleParamsLate.addParameter<std::string>("@module_label", moduleLabelLate);
  moduleParamsLate.registerIt();

  ParameterSet processParamsLate;
  std::string processNameLate("LATE");
  processParamsLate.addParameter<std::string>("@process_name", processNameLate);
  processParamsLate.addParameter(moduleLabelLate, moduleParamsLate);
  processParamsLate.registerIt();

  ProcessConfiguration processLate("LATE", processParamsLate.id(), getReleaseVersion(), getPassID());

  std::auto_ptr<ProcessHistory> processHistory(new ProcessHistory);
  ProcessHistory& ph = *processHistory;
  processHistory->push_back(processEarly);
  processHistory->push_back(processLate);

  ProcessHistoryRegistry::instance()->insertMapped(ph);

  ProcessHistoryID processHistoryID = ph.id();

  // Finally done with making a fake process history

  // This is all a hack to make this test work, but here is some
  // detail as to what is going on.  (this does not happen in real
  // data processing).
  // When any new product is added to the event principal at
  // commit, the "CURRENT" process will go into the ProcessHistory
  // even if we have faked it that the new product is associated
  // with a previous process, because the process name comes from
  // the currentModuleDescription stored in the principal.  On the
  // other hand, when addProduct is called another event is created
  // with a fake moduleDescription containing the old process name
  // and that is used to create the product holder in the principal used to
  // look up the object.

  boost::shared_ptr<ProductRegistry const> preg(availableProducts_);
  std::string uuid = createGlobalIdentifier();
  Timestamp time = make_timestamp();
  EventID id = make_id();
  ProcessConfiguration const& pc = currentModuleDescription_->processConfiguration();
  boost::shared_ptr<RunAuxiliary> runAux(new RunAuxiliary(id.run(), time, time));
  boost::shared_ptr<RunPrincipal> rp(new RunPrincipal(runAux, preg, pc));
  boost::shared_ptr<LuminosityBlockAuxiliary> lumiAux(new LuminosityBlockAuxiliary(rp->run(), 1, time, time));
  boost::shared_ptr<LuminosityBlockPrincipal>lbp(new LuminosityBlockPrincipal(lumiAux, preg, pc));
  lbp->setRunPrincipal(rp);
  EventAuxiliary eventAux(id, uuid, time, true);
  const_cast<ProcessHistoryID &>(eventAux.processHistoryID()) = processHistoryID;
  principal_.reset(new edm::EventPrincipal(preg, branchIDListHelper_, pc));
  principal_->fillEventPrincipal(eventAux);
  principal_->setLuminosityBlockPrincipal(lbp);
  currentEvent_.reset(new Event(*principal_, *currentModuleDescription_));

}

void
testEvent::tearDown() {
  currentEvent_.reset();
  principal_.reset();
}

void testEvent::emptyEvent() {
  CPPUNIT_ASSERT(currentEvent_);
  CPPUNIT_ASSERT(currentEvent_->id() == make_id());
  CPPUNIT_ASSERT(currentEvent_->time() == make_timestamp());
  CPPUNIT_ASSERT(currentEvent_->size() == 0);
}

void testEvent::getByLabelFromEmpty() {
  InputTag inputTag("moduleLabel", "instanceName");
  Handle<int> nonesuch;
  CPPUNIT_ASSERT(!nonesuch.isValid());
  CPPUNIT_ASSERT(!currentEvent_->getByLabel(inputTag, nonesuch));
  CPPUNIT_ASSERT(!nonesuch.isValid());
  CPPUNIT_ASSERT(nonesuch.failedToGet());
  CPPUNIT_ASSERT_THROW(*nonesuch, cms::Exception);
}

void testEvent::putAnIntProduct() {
  std::auto_ptr<edmtest::IntProduct> three(new edmtest::IntProduct(3));
  currentEvent_->put(three, "int1");
  CPPUNIT_ASSERT(currentEvent_->size() == 1);
  EDProducer::commitEvent(*currentEvent_);
  CPPUNIT_ASSERT(currentEvent_->size() == 1);
}

void testEvent::putAndGetAnIntProduct() {
  std::auto_ptr<edmtest::IntProduct> four(new edmtest::IntProduct(4));
  currentEvent_->put(four, "int1");
  EDProducer::commitEvent(*currentEvent_);

  InputTag should_match("modMulti", "int1", "CURRENT");
  InputTag should_not_match("modMulti", "int1", "NONESUCH");
  Handle<edmtest::IntProduct> h;
  currentEvent_->getByLabel(should_match, h);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(!currentEvent_->getByLabel(should_not_match, h));
  CPPUNIT_ASSERT(!h.isValid());
  CPPUNIT_ASSERT_THROW(*h, cms::Exception);
}

void testEvent::getByProductID() {

  typedef edmtest::IntProduct product_t;
  typedef std::auto_ptr<product_t> ap_t;
  typedef Handle<product_t>  handle_t;

  ProductID wanted;

  {
    ap_t one(new product_t(1));
    ProductID id1 = addProduct(one, "int1_tag", "int1");
    CPPUNIT_ASSERT(id1 != ProductID());
    wanted = id1;

    ap_t two(new product_t(2));
    ProductID id2 = addProduct(two, "int2_tag", "int2");
    CPPUNIT_ASSERT(id2 != ProductID());
    CPPUNIT_ASSERT(id2 != id1);

    EDProducer::commitEvent(*currentEvent_);
    CPPUNIT_ASSERT(currentEvent_->size() == 2);
  }

  handle_t h;
  currentEvent_->get(wanted, h);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.id() == wanted);
  CPPUNIT_ASSERT(h->value == 1);

  ProductID invalid;
  CPPUNIT_ASSERT_THROW(currentEvent_->get(invalid, h), cms::Exception);
  CPPUNIT_ASSERT(!h.isValid());
  ProductID notpresent(0, std::numeric_limits<unsigned short>::max());
  CPPUNIT_ASSERT(!currentEvent_->get(notpresent, h));
  CPPUNIT_ASSERT(!h.isValid());
  CPPUNIT_ASSERT(h.failedToGet());
  CPPUNIT_ASSERT_THROW(*h, cms::Exception);
}

void testEvent::transaction() {
  // Put a product into an Event, and make sure that if we don't
  // commit, there is no product in the EventPrincipal afterwards.
  CPPUNIT_ASSERT(principal_->size() == 0);
  {
    typedef edmtest::IntProduct product_t;
    typedef std::auto_ptr<product_t> ap_t;

    ap_t three(new product_t(3));
    currentEvent_->put(three, "int1");
    CPPUNIT_ASSERT(principal_->size() == 0);
    CPPUNIT_ASSERT(currentEvent_->size() == 1);
    // DO NOT COMMIT!
  }

  // The Event has been destroyed without a commit -- we should not
  // have any products in the EventPrincipal.
  CPPUNIT_ASSERT(principal_->size() == 0);
}

void testEvent::getByLabel() {
  typedef edmtest::IntProduct product_t;
  typedef std::auto_ptr<product_t> ap_t;
  typedef Handle<product_t> handle_t;
  typedef std::vector<handle_t> handle_vec;

  ap_t one(new product_t(1));
  ap_t two(new product_t(2));
  ap_t three(new product_t(3));
  ap_t four(new product_t(4));
  addProduct(one,   "int1_tag", "int1");
  addProduct(two,   "int2_tag", "int2");
  addProduct(three, "int3_tag");
  addProduct(four,  "nolabel_tag");

  std::auto_ptr<std::vector<edmtest::Thing> > ap_vthing(new std::vector<edmtest::Thing>);
  addProduct(ap_vthing, "thing", "");

  ap_t oneHundred(new product_t(100));
  addProduct(oneHundred, "int1_tag_late", "int1");

  std::auto_ptr<edmtest::IntProduct> twoHundred(new edmtest::IntProduct(200));
  currentEvent_->put(twoHundred, "int1");
  EDProducer::commitEvent(*currentEvent_);

  CPPUNIT_ASSERT(currentEvent_->size() == 7);

  handle_t h;
  CPPUNIT_ASSERT(currentEvent_->getByLabel("modMulti", h));
  CPPUNIT_ASSERT(h->value == 3);

  CPPUNIT_ASSERT(currentEvent_->getByLabel("modMulti", "int1", h));
  CPPUNIT_ASSERT(h->value == 200);

  CPPUNIT_ASSERT(!currentEvent_->getByLabel("modMulti", "nomatch", h));
  CPPUNIT_ASSERT(!h.isValid());
  CPPUNIT_ASSERT_THROW(*h, cms::Exception);

  InputTag inputTag("modMulti", "int1");
  CPPUNIT_ASSERT(currentEvent_->getByLabel(inputTag, h));
  CPPUNIT_ASSERT(h->value == 200);
  {
    handle_t h;
    edm::EventBase* baseEvent = currentEvent_.get();
    CPPUNIT_ASSERT(baseEvent->getByLabel(inputTag, h));
    CPPUNIT_ASSERT(h->value == 200);

  }

  size_t cachedOffset = 0;
  int fillCount = -1;

  BasicHandle bh = principal_->getByLabel(TypeID(typeid(edmtest::IntProduct)), "modMulti", "int1", "LATE", cachedOffset, fillCount);
  convert_handle(bh, h);
  CPPUNIT_ASSERT(h->value == 100);
  BasicHandle bh2(principal_->getByLabel(TypeID(typeid(edmtest::IntProduct)), "modMulti", "int1", "nomatch", cachedOffset, fillCount));
  CPPUNIT_ASSERT(!bh2.isValid());

  boost::shared_ptr<Wrapper<edmtest::IntProduct> const> ptr = getProductByTag<edmtest::IntProduct>(*principal_, inputTag);
  CPPUNIT_ASSERT(ptr->product()->value == 200);
}

void testEvent::getManyByType() {
  typedef edmtest::IntProduct product_t;
  typedef std::auto_ptr<product_t> ap_t;
  typedef Handle<product_t> handle_t;
  typedef std::vector<handle_t> handle_vec;

  ap_t one(new product_t(1));
  ap_t two(new product_t(2));
  ap_t three(new product_t(3));
  ap_t four(new product_t(4));
  addProduct(one,   "int1_tag", "int1");
  addProduct(two,   "int2_tag", "int2");
  addProduct(three, "int3_tag");
  addProduct(four,  "nolabel_tag");

  std::auto_ptr<std::vector<edmtest::Thing> > ap_vthing(new std::vector<edmtest::Thing>);
  addProduct(ap_vthing, "thing", "");

  std::auto_ptr<std::vector<edmtest::Thing> > ap_vthing2(new std::vector<edmtest::Thing>);
  addProduct(ap_vthing2, "thing2", "inst2");

  ap_t oneHundred(new product_t(100));
  addProduct(oneHundred, "int1_tag_late", "int1");

  std::auto_ptr<edmtest::IntProduct> twoHundred(new edmtest::IntProduct(200));
  currentEvent_->put(twoHundred, "int1");
  EDProducer::commitEvent(*currentEvent_);

  CPPUNIT_ASSERT(currentEvent_->size() == 8);

  handle_vec handles;
  currentEvent_->getManyByType(handles);
  CPPUNIT_ASSERT(handles.size() == 6);
  int sum = 0;
  for (int k = 0; k < 6; ++k) sum += handles[k]->value;
  CPPUNIT_ASSERT(sum == 310);
}

void testEvent::printHistory() {
  ProcessHistory const& history = currentEvent_->processHistory();
  std::ofstream out("history.log");

  copy_all(history, std::ostream_iterator<ProcessHistory::const_iterator::value_type>(out, "\n"));
}

void testEvent::deleteProduct() {
  
  typedef edmtest::IntProduct product_t;
  typedef std::auto_ptr<product_t> ap_t;
  typedef Handle<product_t> handle_t;
  
  ap_t one(new product_t(1));
  addProduct(one,   "int1_tag", "int1");
  
  BranchID id;
  
  availableProducts_->callForEachBranch([&id](const BranchDescription& iDesc){ 
    if(iDesc.moduleLabel()=="modMulti" && iDesc.productInstanceName()=="int1") {
      id = iDesc.branchID();
    }});

  const ProductHolderBase* phb = principal_->getProductHolder(id,false,false);
  CPPUNIT_ASSERT(phb != nullptr);
  
  CPPUNIT_ASSERT(!phb->productWasDeleted());  
  principal_->deleteProduct(id);
  CPPUNIT_ASSERT(phb->productWasDeleted());
  
}
