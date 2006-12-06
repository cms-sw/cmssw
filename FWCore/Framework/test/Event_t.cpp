/*----------------------------------------------------------------------

Test program for edm::Event.

$Id: Event_t.cpp,v 1.2 2006/12/05 23:56:18 paterno Exp $
----------------------------------------------------------------------*/
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#include <memory>

#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "FWCore/Utilities/interface/GetPassID.h"

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
  CPPUNIT_TEST_SUITE_END();

 public:
  testEvent();
  void setUp();
  void tearDown();
  void emptyEvent();
  void getBySelectorFromEmpty();
  void putAnIntProduct();
  void putAndGetAnIntProduct();

 private:
  
  ProductRegistry*   availableProducts_;
  EventPrincipal*    principal_;
  Event*             currentEvent_;
  ModuleDescription* currentModuleDescription_;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEvent);

namespace
{
  template <class T> void kill_and_clear(T*& p) { delete p; p=0; }
}

EventID   make_id() { return EventID(2112, 25, true); }
Timestamp make_timestamp() { return Timestamp(1); }

testEvent::testEvent() :
  availableProducts_(0),
  principal_(0),
  currentEvent_(0),
  currentModuleDescription_(0)
{ }

void
testEvent::setUp()
{
  availableProducts_ = new ProductRegistry();

  // Fake up the production of a single IntProduct from an IntProducer
  // module, run in the 'FUNKY' process.

  ParameterSet moduleParams;
  string moduleLabel("modInt");
  string moduleClassName("IntProducer");
  moduleParams.addParameter<std::string>("@module_type", moduleClassName);
  moduleParams.addParameter<std::string>("@module_label", moduleLabel);

  ParameterSet processParams;
  string processName("FUNKY");
  processParams.addParameter<std::string>("@process_name", processName);
  processParams.addParameter(moduleLabel, moduleParams);

  ProcessConfiguration process;
  process.processName_    = processName;
  process.releaseVersion_ = getReleaseVersion();
  process.passID_         = getPassID();
  process.parameterSetID_ = processParams.id();

  TypeID product_type(typeid(edmtest::IntProduct));

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
  availableProducts_->setProductIDs();
  availableProducts_->setFrozen();
    
  principal_  = new EventPrincipal(make_id(),
				   make_timestamp(),
				   *availableProducts_);

  currentEvent_ = new Event(*principal_, *currentModuleDescription_);
}

void
testEvent::tearDown()
{
  kill_and_clear(currentEvent_);
  kill_and_clear(currentModuleDescription_);
  kill_and_clear(principal_);
  kill_and_clear(availableProducts_);  
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

  ProcessNameSelector should_match("FUNKY");
  ProcessNameSelector should_not_match("FUN");
  Handle<edmtest::IntProduct> h;
  currentEvent_->get(should_match, h);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT_THROW(currentEvent_->get(should_not_match, h), edm::Exception);
  CPPUNIT_ASSERT(!h.isValid());
}

