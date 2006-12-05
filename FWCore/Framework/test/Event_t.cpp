/*----------------------------------------------------------------------

Test program for edm::Event.

$Id: Event_t.cpp,v 1.1 2006/11/30 15:37:07 paterno Exp $
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

class testEvent: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testEvent);
  CPPUNIT_TEST(emptyEvent);
  CPPUNIT_TEST(getBySelectorFromEmpty);
  CPPUNIT_TEST(putAnIntProduct);
  CPPUNIT_TEST_SUITE_END();

 public:
  testEvent();
  void setUp();
  void tearDown();
  void emptyEvent();
  void getBySelectorFromEmpty();
  void putAnIntProduct();

 private:
  
  ProductRegistry*   availableProducts_;
  EventPrincipal*    principal_;
  Event*             emptyEvent_;
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
  emptyEvent_(0),
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

  emptyEvent_ = new Event(*principal_, *currentModuleDescription_);
}

void
testEvent::tearDown()
{
  kill_and_clear(emptyEvent_);
  kill_and_clear(currentModuleDescription_);
  kill_and_clear(principal_);
  kill_and_clear(availableProducts_);  
}

void testEvent::emptyEvent()

{
  CPPUNIT_ASSERT(emptyEvent_);
  CPPUNIT_ASSERT(emptyEvent_->id() == make_id());
  CPPUNIT_ASSERT(emptyEvent_->time() == make_timestamp());
  CPPUNIT_ASSERT(emptyEvent_->size() == 0);  
}

void testEvent::getBySelectorFromEmpty()
{
  ModuleLabelSelector byModuleLabel("mod1");
  Handle<int> nonesuch;
  CPPUNIT_ASSERT(!nonesuch.isValid());
  CPPUNIT_ASSERT_THROW(emptyEvent_->get(byModuleLabel, nonesuch),
		       edm::Exception);
}

void testEvent::putAnIntProduct()
{
  auto_ptr<edmtest::IntProduct> three(new edmtest::IntProduct(3));
  emptyEvent_->put(three);
  CPPUNIT_ASSERT(emptyEvent_->size() == 1);
}

