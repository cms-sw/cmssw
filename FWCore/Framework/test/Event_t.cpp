/*----------------------------------------------------------------------

Test program for edm::Event.

$Id: typeid_t.cppunit.cc,v 1.3 2006/02/20 01:51:59 wmtan Exp $
----------------------------------------------------------------------*/
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#include <memory>

#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"

using namespace edm;
using namespace std;

class testEvent: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testEvent);
  CPPUNIT_TEST(emptyEvent);
  CPPUNIT_TEST(getBySelectorFromEmpty);
  //  CPPUNIT_TEST(putAnInt);
  CPPUNIT_TEST_SUITE_END();

 public:
  void setUp();
  void tearDown();
  void emptyEvent();
  void getBySelectorFromEmpty();
  //void putAnInt();

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

void
testEvent::setUp()
{
  availableProducts_ = new ProductRegistry();

  BranchDescription intDescription;
  availableProducts_->addProduct(intDescription);
  availableProducts_->setProductIDs();
  availableProducts_->setFrozen();
    
  principal_  = new EventPrincipal(make_id(),
				   make_timestamp(),
				   *availableProducts_);
  currentModuleDescription_ = new ModuleDescription();
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

// void testEvent::putAnInt()
// {
//   auto_ptr<int> three(new int(3));
//   emptyEvent_->put(three);
//   CPPUNIT_ASSERT(emptyEvent_->size() == 1);
// }

