
/*----------------------------------------------------------------------

Test program for edm::Event.

$Id: Event_t.cpp,v 1.15 2007/06/21 16:52:43 wmtan Exp $
----------------------------------------------------------------------*/
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <typeinfo>
#include <vector>

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

using namespace edm;
using namespace std;

// This is a gross hack, to allow us to test the event
namespace edm
{
  class ProducerWorker
  {
  public:
    static void commitEvent(Event& e) { e.commit_(); }
    
  };
}


class testEvent: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testEvent);
  CPPUNIT_TEST(emptyEvent);
  CPPUNIT_TEST(getBySelectorFromEmpty);
  CPPUNIT_TEST(putAnIntProduct);
  CPPUNIT_TEST(putAndGetAnIntProduct);
  CPPUNIT_TEST(getByProductID);
  CPPUNIT_TEST(transaction);
  CPPUNIT_TEST(getByInstanceName);
  CPPUNIT_TEST(getBySelector);
  CPPUNIT_TEST(getByLabel);
  CPPUNIT_TEST(getByType);
  CPPUNIT_TEST(printHistory);
  CPPUNIT_TEST_SUITE_END();

 public:
  testEvent();
  ~testEvent();
  void setUp();
  void tearDown();
  void emptyEvent();
  void getBySelectorFromEmpty();
  void putAnIntProduct();
  void putAndGetAnIntProduct();
  void getByProductID();
  void transaction();
  void getByInstanceName();
  void getBySelector();
  void getByLabel();
  void getByType();
  void printHistory();

 private:

  template <class T>
  void registerProduct(string const& tag,
		       string const& moduleLabel,
		       string const& moduleClassName,
		       string const& processName,
		       string const& productInstanceName);

  template <class T>
  void registerProduct(string const& tag,
		       string const& moduleLabel,
		       string const& moduleClassName,
		       string const& processName)
  {
    string productInstanceName;
    registerProduct<T>(tag, moduleLabel, moduleClassName, processName, 
		       productInstanceName);
  }

  template <class T>
  ProductID addProduct(auto_ptr<T> product,
		       string const& tag,
		       string const& productLabel = string());
  
  ProductRegistry*   availableProducts_;
  EventPrincipal*    principal_;
  Event*             currentEvent_;
  ModuleDescription* currentModuleDescription_;
  typedef map<string, ModuleDescription> modCache_t;
  typedef modCache_t::iterator iterator_t;

  modCache_t moduleDescriptions_;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEvent);

namespace
{
  template <class T> void kill_and_clear(T*& p) { delete p; p = 0; }
}

EventID   make_id() { return EventID(2112, 25); }
Timestamp make_timestamp() { return Timestamp(1); }

template <class T>
void
testEvent::registerProduct(string const& tag,
			   string const& moduleLabel,
 			   string const& moduleClassName,
 			   string const& processName,
			   string const& productInstanceName)
{
  if (!availableProducts_)
    availableProducts_ = new ProductRegistry();
  
  ParameterSet moduleParams;
  moduleParams.template addParameter<string>("@module_type", moduleClassName);
  moduleParams.template addParameter<string>("@module_label", moduleLabel);
  
  ParameterSet processParams;
  processParams.template addParameter<string>("@process_name", processName);
  processParams.template addParameter<ParameterSet>(moduleLabel, moduleParams);
  
  ProcessConfiguration process;
  process.processName_    = processName;
  process.releaseVersion_ = getReleaseVersion();
  process.passID_         = getPassID();
  process.parameterSetID_ = processParams.id();

  ModuleDescription localModuleDescription;
  localModuleDescription.parameterSetID_       = moduleParams.id();
  localModuleDescription.moduleName_           = moduleClassName;
  localModuleDescription.moduleLabel_          = moduleLabel;
  localModuleDescription.processConfiguration_ = process;
  
  BranchDescription branch;

  TypeID product_type(typeid(T));
  branch.moduleLabel_         = moduleLabel;
  branch.processName_         = processName;
  branch.productInstanceName_ = productInstanceName;
  branch.fullClassName_       = product_type.userClassName();
  branch.friendlyClassName_   = product_type.friendlyClassName();
  branch.moduleDescriptionID_ = localModuleDescription.id();
  
  moduleDescriptions_[tag] = localModuleDescription;
  availableProducts_->addProduct(branch);
}

// Add the given product, of type T, to the current event,
// as if it came from the module specified by the given tag.
template <class T>
ProductID
testEvent::addProduct(auto_ptr<T> product,
		      string const& tag,
		      string const& productLabel)
{
  iterator_t description = moduleDescriptions_.find(tag);
  if (description == moduleDescriptions_.end())
    throw edm::Exception(errors::LogicError)
      << "Failed to find a module description for tag: " 
      << tag << '\n';

  Event temporaryEvent(*principal_, description->second);
  OrphanHandle<T> h = temporaryEvent.put(product, productLabel);
  ProductID id = h.id();
  ProducerWorker::commitEvent(temporaryEvent);
  return id;
}

testEvent::testEvent() :
  availableProducts_(new ProductRegistry()),
  principal_(0),
  currentEvent_(0),
  currentModuleDescription_(0),
  moduleDescriptions_()
{
  typedef edmtest::IntProduct prod_t;
  typedef vector<edmtest::Thing> vec_t;

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
  string moduleLabel("modMulti");
  string moduleClassName("IntProducer");
  moduleParams.addParameter<string>("@module_type", moduleClassName);
  moduleParams.addParameter<string>("@module_label", moduleLabel);

  ParameterSet processParams;
  string processName("CURRENT");
  processParams.addParameter<string>("@process_name", processName);
  processParams.addParameter(moduleLabel, moduleParams);

  ProcessConfiguration process;
  process.processName_    = processName;
  process.releaseVersion_ = getReleaseVersion();
  process.passID_         = getPassID();
  process.parameterSetID_ = processParams.id();

  TypeID product_type(typeid(prod_t));

  currentModuleDescription_ = new ModuleDescription();
  currentModuleDescription_->parameterSetID_       = moduleParams.id();
  currentModuleDescription_->moduleName_           = moduleClassName;
  currentModuleDescription_->moduleLabel_          = moduleLabel;
  currentModuleDescription_->processConfiguration_ = process;

  string productInstanceName("int1");

  BranchDescription branch;
  branch.moduleLabel_         = moduleLabel;
  branch.productInstanceName_ = productInstanceName;
  branch.processName_         = processName;
  branch.fullClassName_       = product_type.userClassName();
  branch.friendlyClassName_   = product_type.friendlyClassName();
  branch.moduleDescriptionID_ = currentModuleDescription_->id();
  
  availableProducts_->addProduct(branch);


  // Freeze the product registry before we make the Event.
  availableProducts_->setProductIDs();
  availableProducts_->setFrozen();
}

testEvent::~testEvent()
{
  delete principal_;
  delete currentEvent_;
  delete currentModuleDescription_;
}

void testEvent::setUp() 
{

  // First build a fake process history, that says there
  // were previous processes named "EARLY" and "LATE".
  // This takes several lines of code but other than
  // the process names none of it is used or interesting.
  ParameterSet moduleParamsEarly;
  string moduleLabelEarly("currentModule");
  string moduleClassNameEarly("IntProducer");
  moduleParamsEarly.addParameter<string>("@module_type", moduleClassNameEarly);
  moduleParamsEarly.addParameter<string>("@module_label", moduleLabelEarly);

  ParameterSet processParamsEarly;
  string processNameEarly("EARLY");
  processParamsEarly.addParameter<string>("@process_name", processNameEarly);
  processParamsEarly.addParameter(moduleLabelEarly, moduleParamsEarly);

  ProcessConfiguration processEarly;
  processEarly.processName_    = "EARLY";
  processEarly.releaseVersion_ = getReleaseVersion();
  processEarly.passID_         = getPassID();
  processEarly.parameterSetID_ = processParamsEarly.id();

  ParameterSet moduleParamsLate;
  string moduleLabelLate("currentModule");
  string moduleClassNameLate("IntProducer");
  moduleParamsLate.addParameter<string>("@module_type", moduleClassNameLate);
  moduleParamsLate.addParameter<string>("@module_label", moduleLabelLate);

  ParameterSet processParamsLate;
  string processNameLate("LATE");
  processParamsLate.addParameter<string>("@process_name", processNameLate);
  processParamsLate.addParameter(moduleLabelLate, moduleParamsLate);

  ProcessConfiguration processLate;
  processLate.processName_    = "LATE";
  processLate.releaseVersion_ = getReleaseVersion();
  processLate.passID_         = getPassID();
  processLate.parameterSetID_ = processParamsLate.id();

  ProcessHistory* processHistory = new ProcessHistory;
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
  // and that is used to create the group in the principal used to
  // look up the object.

  boost::shared_ptr<ProductRegistry const> preg = boost::shared_ptr<ProductRegistry const>(availableProducts_);
  principal_  = new EventPrincipal(make_id(),
				   make_timestamp(),
				   preg,
                                   1,
                                   currentModuleDescription_->processConfiguration(),
                                   true,
				   std::string("Unspecified"),
                                   processHistoryID);

  currentEvent_ = new Event(*principal_, *currentModuleDescription_);

  delete processHistory;
}

void
testEvent::tearDown()
{
  kill_and_clear(currentEvent_);
  kill_and_clear(principal_);
}

void testEvent::emptyEvent()

{
  CPPUNIT_ASSERT(currentEvent_);
  CPPUNIT_ASSERT(currentEvent_->id() == make_id());
  CPPUNIT_ASSERT(currentEvent_->time() == make_timestamp());
  CPPUNIT_ASSERT(currentEvent_->size() == 0);  
}

void testEvent::getBySelectorFromEmpty()
{
  ModuleLabelSelector byModuleLabel("mod1");
  Handle<int> nonesuch;
  CPPUNIT_ASSERT(!nonesuch.isValid());
  CPPUNIT_ASSERT_THROW(currentEvent_->get(byModuleLabel, nonesuch),
		       edm::Exception);
}

void testEvent::putAnIntProduct()
{
  auto_ptr<edmtest::IntProduct> three(new edmtest::IntProduct(3));
  currentEvent_->put(three, "int1");
  CPPUNIT_ASSERT(currentEvent_->size() == 1);
  ProducerWorker::commitEvent(*currentEvent_);
  CPPUNIT_ASSERT(currentEvent_->size() == 1);
}

void testEvent::putAndGetAnIntProduct()
{
  auto_ptr<edmtest::IntProduct> four(new edmtest::IntProduct(4));
  currentEvent_->put(four, "int1");
  ProducerWorker::commitEvent(*currentEvent_);

  ProcessNameSelector should_match("CURRENT");
  ProcessNameSelector should_not_match("NONESUCH");
  Handle<edmtest::IntProduct> h;
  currentEvent_->get(should_match, h);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT_THROW(currentEvent_->get(should_not_match, h), edm::Exception);
  CPPUNIT_ASSERT(!h.isValid());
}

void testEvent::getByProductID()
{
  
  typedef edmtest::IntProduct product_t;
  typedef auto_ptr<product_t> ap_t;
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
    
    ProducerWorker::commitEvent(*currentEvent_);
    CPPUNIT_ASSERT(currentEvent_->size() == 2);
  }

  handle_t h;
  currentEvent_->get(wanted, h);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.id() == wanted);
  CPPUNIT_ASSERT(h->value == 1);

  ProductID invalid;
  CPPUNIT_ASSERT_THROW(currentEvent_->get(invalid, h), edm::Exception);
  CPPUNIT_ASSERT(!h.isValid());
  ProductID notpresent(std::numeric_limits<unsigned int>::max());
  CPPUNIT_ASSERT_THROW(currentEvent_->get(notpresent, h), edm::Exception);
  CPPUNIT_ASSERT(!h.isValid());
}

void testEvent::transaction()
{
  // Put a product into an Event, and make sure that if we don't
  // commit, there is no product in the EventPrincipal afterwards.
  CPPUNIT_ASSERT(principal_->size() == 0);
  {
    typedef edmtest::IntProduct product_t;
    typedef auto_ptr<product_t> ap_t;

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

void testEvent::getByInstanceName()
{
  typedef edmtest::IntProduct product_t;
  typedef auto_ptr<product_t> ap_t;
  typedef Handle<product_t> handle_t;
  typedef vector<handle_t> handle_vec;

  ap_t one(new product_t(1));
  ap_t two(new product_t(2));
  ap_t three(new product_t(3));
  ap_t four(new product_t(4));
  addProduct(one,   "int1_tag", "int1");
  addProduct(two,   "int2_tag", "int2");
  addProduct(three, "int3_tag");
  addProduct(four,  "nolabel_tag");

  CPPUNIT_ASSERT(currentEvent_->size() == 4);

  Selector sel(ProductInstanceNameSelector("int2") &&
	       ModuleLabelSelector("modMulti"));;
  handle_t h;
  currentEvent_->get(sel, h);
  CPPUNIT_ASSERT(h->value == 2);

  string instance;
  Selector sel1(ProductInstanceNameSelector(instance) &&
	       ModuleLabelSelector("modMulti"));;

  currentEvent_->get(sel1, h);
  CPPUNIT_ASSERT(h->value == 3);

  handle_vec handles;
  currentEvent_->getMany(ModuleLabelSelector("modMulti"), handles);
  CPPUNIT_ASSERT(handles.size() == 3);
  handles.clear();
  currentEvent_->getMany(ModuleLabelSelector("nomatch"), handles);
  CPPUNIT_ASSERT(handles.empty());
  vector<Handle<int> > nomatches;
  currentEvent_->getMany(ModuleLabelSelector("modMulti"), nomatches);
  CPPUNIT_ASSERT(nomatches.empty());
}

void testEvent::getBySelector()
{
  typedef edmtest::IntProduct product_t;
  typedef auto_ptr<product_t> ap_t;
  typedef Handle<product_t> handle_t;
  typedef vector<handle_t> handle_vec;

  ap_t one(new product_t(1));
  ap_t two(new product_t(2));
  ap_t three(new product_t(3));
  ap_t four(new product_t(4));
  addProduct(one,   "int1_tag", "int1");
  addProduct(two,   "int2_tag", "int2");
  addProduct(three, "int3_tag");
  addProduct(four,  "nolabel_tag");

  auto_ptr<vector<edmtest::Thing> > ap_vthing(new vector<edmtest::Thing>);
  addProduct(ap_vthing, "thing", "");

  ap_t oneHundred(new product_t(100));
  addProduct(oneHundred, "int1_tag_late", "int1");

  auto_ptr<edmtest::IntProduct> twoHundred(new edmtest::IntProduct(200));
  currentEvent_->put(twoHundred, "int1");
  ProducerWorker::commitEvent(*currentEvent_);

  CPPUNIT_ASSERT(currentEvent_->size() == 7);

  Selector sel(ProductInstanceNameSelector("int2") &&
	       ModuleLabelSelector("modMulti") &&
               ProcessNameSelector("EARLY"));;
  handle_t h;
  currentEvent_->get(sel, h);
  CPPUNIT_ASSERT(h->value == 2);

  Selector sel1(ProductInstanceNameSelector("nomatch") &&
	        ModuleLabelSelector("modMulti") &&
                ProcessNameSelector("EARLY"));;
  CPPUNIT_ASSERT_THROW(currentEvent_->get(sel1, h), edm::Exception);

  Selector sel2(ProductInstanceNameSelector("int2") &&
	        ModuleLabelSelector("nomatch") &&
                ProcessNameSelector("EARLY"));;
  CPPUNIT_ASSERT_THROW(currentEvent_->get(sel2, h), edm::Exception);

  Selector sel3(ProductInstanceNameSelector("int2") &&
	        ModuleLabelSelector("modMulti") &&
                ProcessNameSelector("nomatch"));;
  CPPUNIT_ASSERT_THROW(currentEvent_->get(sel3, h), edm::Exception);

  Selector sel4(ModuleLabelSelector("modMulti") &&
                ProcessNameSelector("EARLY"));;
  CPPUNIT_ASSERT_THROW(currentEvent_->get(sel4, h), edm::Exception);

  Selector sel5(ModuleLabelSelector("modMulti") &&
                ProcessNameSelector("LATE"));;
  currentEvent_->get(sel5, h);
  CPPUNIT_ASSERT(h->value == 100);

  Selector sel6(ModuleLabelSelector("modMulti") &&
                ProcessNameSelector("CURRENT"));;
  currentEvent_->get(sel6, h);
  CPPUNIT_ASSERT(h->value == 200);

  Selector sel7(ModuleLabelSelector("modMulti"));;
  currentEvent_->get(sel7, h);
  CPPUNIT_ASSERT(h->value == 200);

  handle_vec handles;
  currentEvent_->getMany(ModuleLabelSelector("modMulti"), handles);
  CPPUNIT_ASSERT(handles.size() == 5);
  int sum = 0;
  for (int k = 0; k < 5; ++k) sum += handles[k]->value;
  CPPUNIT_ASSERT(sum == 306);
}

void testEvent::getByLabel()
{
  typedef edmtest::IntProduct product_t;
  typedef auto_ptr<product_t> ap_t;
  typedef Handle<product_t> handle_t;
  typedef vector<handle_t> handle_vec;

  ap_t one(new product_t(1));
  ap_t two(new product_t(2));
  ap_t three(new product_t(3));
  ap_t four(new product_t(4));
  addProduct(one,   "int1_tag", "int1");
  addProduct(two,   "int2_tag", "int2");
  addProduct(three, "int3_tag");
  addProduct(four,  "nolabel_tag");

  auto_ptr<vector<edmtest::Thing> > ap_vthing(new vector<edmtest::Thing>);
  addProduct(ap_vthing, "thing", "");

  ap_t oneHundred(new product_t(100));
  addProduct(oneHundred, "int1_tag_late", "int1");

  auto_ptr<edmtest::IntProduct> twoHundred(new edmtest::IntProduct(200));
  currentEvent_->put(twoHundred, "int1");
  ProducerWorker::commitEvent(*currentEvent_);

  CPPUNIT_ASSERT(currentEvent_->size() == 7);

  handle_t h;
  currentEvent_->getByLabel("modMulti", h);
  CPPUNIT_ASSERT(h->value == 3);

  currentEvent_->getByLabel("modMulti", "int1", h);
  CPPUNIT_ASSERT(h->value == 200);

  CPPUNIT_ASSERT_THROW(currentEvent_->getByLabel("modMulti", "nomatch", h), edm::Exception);

  InputTag inputTag("modMulti", "int1");
  currentEvent_->getByLabel(inputTag, h);
  CPPUNIT_ASSERT(h->value == 200);

  BasicHandle bh =
    principal_->getByLabel(TypeID(typeid(edmtest::IntProduct)), "modMulti", "int1", "LATE");
  convert_handle(bh, h);
  CPPUNIT_ASSERT(h->value == 100);
  CPPUNIT_ASSERT_THROW(principal_->getByLabel(TypeID(typeid(edmtest::IntProduct)), "modMulti", "int1", "nomatch"), edm::Exception);
}


void testEvent::getByType()
{
  typedef edmtest::IntProduct product_t;
  typedef auto_ptr<product_t> ap_t;
  typedef Handle<product_t> handle_t;
  typedef vector<handle_t> handle_vec;

  ap_t one(new product_t(1));
  ap_t two(new product_t(2));
  ap_t three(new product_t(3));
  ap_t four(new product_t(4));
  addProduct(one,   "int1_tag", "int1");
  addProduct(two,   "int2_tag", "int2");
  addProduct(three, "int3_tag");
  addProduct(four,  "nolabel_tag");

  auto_ptr<vector<edmtest::Thing> > ap_vthing(new vector<edmtest::Thing>);
  addProduct(ap_vthing, "thing", "");

  auto_ptr<vector<edmtest::Thing> > ap_vthing2(new vector<edmtest::Thing>);
  addProduct(ap_vthing2, "thing2", "inst2");

  ap_t oneHundred(new product_t(100));
  addProduct(oneHundred, "int1_tag_late", "int1");

  auto_ptr<edmtest::IntProduct> twoHundred(new edmtest::IntProduct(200));
  currentEvent_->put(twoHundred, "int1");
  ProducerWorker::commitEvent(*currentEvent_);

  CPPUNIT_ASSERT(currentEvent_->size() == 8);

  handle_t h;
  currentEvent_->getByType(h);
  CPPUNIT_ASSERT(h->value == 200);

  Handle<int> h_nomatch;
  CPPUNIT_ASSERT_THROW(currentEvent_->getByType(h_nomatch), edm::Exception);

  Handle<vector<edmtest::Thing> > hthing;
  CPPUNIT_ASSERT_THROW(currentEvent_->getByType(hthing), edm::Exception);

  handle_vec handles;
  currentEvent_->getManyByType(handles);
  CPPUNIT_ASSERT(handles.size() == 6);
  int sum = 0;
  for (int k = 0; k < 6; ++k) sum += handles[k]->value;
  CPPUNIT_ASSERT(sum == 310);
}

void testEvent::printHistory()
{
  ProcessHistory const& history = currentEvent_->processHistory();
  ofstream out("history.log");
  
  copy(history.begin(), history.end(),
       ostream_iterator<ProcessHistory::const_iterator::value_type>(out, "\n"));
}
