// NOTE: Process history is not currently being filled in. Thus any
// testing that expects results to be returned based upon correct
// handling of processing history will not work.

/*----------------------------------------------------------------------

Test program for edm::Event.

$Id: Event_t.cpp,v 1.6 2006/12/18 06:00:21 paterno Exp $
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

#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/ProcessHistory.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/OrphanHandle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"


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
  CPPUNIT_TEST(getViewBySelector);
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
  void getViewBySelector();

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
  template <class T> void kill_and_clear(T*& p) { delete p; p=0; }
}

EventID   make_id() { return EventID(2112, 25, true); }
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

  registerProduct<prod_t>("nolabel_tag", "modOne",   "IntProducer", "EARLY");
  registerProduct<prod_t>("int1_tag",    "modMulti", "IntProducer", "EARLY", "int1");
  registerProduct<prod_t>("int2_tag",    "modMulti", "IntProducer", "EARLY", "int2");
  registerProduct<prod_t>("int3_tag",    "modMulti", "IntProducer", "EARLY", "int3");
  registerProduct<vec_t>("thing",        "modthing", "ThingProducer", "LATE");

  // Fake up the production of a single IntProduct from an IntProducer
  // module, run in the 'CURRENT' process.
  ParameterSet moduleParams;
  string moduleLabel("currentModule");
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
  currentModuleDescription_ = new ModuleDescription();
  currentModuleDescription_->parameterSetID_       = moduleParams.id();
  currentModuleDescription_->moduleName_           = moduleClassName;
  currentModuleDescription_->moduleLabel_          = moduleLabel;
  currentModuleDescription_->processConfiguration_ = process;

  BranchDescription branch;
  branch.moduleLabel_         = moduleLabel;
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
  delete availableProducts_;
  delete principal_;
  delete currentEvent_;
  delete currentModuleDescription_;
}

void testEvent::setUp() 
{
  principal_  = new EventPrincipal(make_id(),
				   make_timestamp(),
				   *availableProducts_);

  currentEvent_ = new Event(*principal_, *currentModuleDescription_);
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
  currentEvent_->put(three);
  CPPUNIT_ASSERT(currentEvent_->size() == 1);
  ProducerWorker::commitEvent(*currentEvent_);
  CPPUNIT_ASSERT(currentEvent_->size() == 1);
}

void testEvent::putAndGetAnIntProduct()
{
  auto_ptr<edmtest::IntProduct> four(new edmtest::IntProduct(4));
  currentEvent_->put(four);
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
    CPPUNIT_ASSERT(id2 != id1 );
    
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
  CPPUNIT_ASSERT( principal_->size() == 0 );
  {
    typedef edmtest::IntProduct product_t;
    typedef auto_ptr<product_t> ap_t;

    ap_t three(new product_t(3));
    currentEvent_->put(three);
    CPPUNIT_ASSERT( principal_->size() == 0 );
    CPPUNIT_ASSERT( currentEvent_->size() == 1);
    // DO NOT COMMIT!
  }

  // The Event has been destroyed without a commit -- we should not
  // have any products in the EventPrincipal.
  CPPUNIT_ASSERT( principal_->size() == 0 );  
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
  addProduct(three, "int3_tag", "int3");
  addProduct(four,  "nolabel_tag");

  CPPUNIT_ASSERT(currentEvent_->size() == 4);

  Selector sel(ProductInstanceNameSelector("int2") &&
	       ModuleLabelSelector("modMulti"));;
  handle_t h;
  currentEvent_->get(sel, h);
  CPPUNIT_ASSERT(h->value == 2);

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

void testEvent::getViewBySelector()
{
  ProcessHistory const& history = currentEvent_->processHistory();
  ofstream out("history.log");
  
  copy(history.begin(), history.end(),
       ostream_iterator<ProcessHistory::const_iterator::value_type>(out, "\n"));
}
