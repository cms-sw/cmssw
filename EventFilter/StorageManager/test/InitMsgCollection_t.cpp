#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/test/TestHelper.h"


/////////////////////////////////////////////////////////////
//
// This test exercises the InitMsgCollection class
//
/////////////////////////////////////////////////////////////

using namespace stor;

using stor::testhelper::allocate_frame_with_init_msg;

class testInitMsgCollection : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testInitMsgCollection);
  CPPUNIT_TEST(testAdditions);

  CPPUNIT_TEST_SUITE_END();

public:
  void testAdditions();

private:
  boost::shared_ptr<InitMsgCollection> initMsgCollection_;
};


void testInitMsgCollection::testAdditions()
{
  using toolbox::mem::Reference;
  if (initMsgCollection_.get() == 0)
    {
      initMsgCollection_.reset(new InitMsgCollection());
    }
  CPPUNIT_ASSERT(initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("hltOutputDQM").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("HLTDEBUG").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("CALIB").get() == 0);

  // *** first INIT message ***

  Reference* ref = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  InitMsgSharedPtr serializedProds;
  CPPUNIT_ASSERT( initMsgCollection_->addIfUnique(initMsgFrag,serializedProds) );
  InitMsgView initMsgView(&(*serializedProds)[0]);
  CPPUNIT_ASSERT(initMsgView.outputModuleLabel() == "hltOutputDQM");
  CPPUNIT_ASSERT(initMsgView.size() == serializedProds->size());
  
  CPPUNIT_ASSERT(initMsgCollection_->size() == 1);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("hltOutputDQM").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("HLTDEBUG").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("CALIB").get() == 0);

  // *** second INIT message ***

  ref = allocate_frame_with_init_msg("HLTDEBUG");
  stor::I2OChain initMsgFrag2(ref);
  CPPUNIT_ASSERT(initMsgFrag2.messageCode() == Header::INIT);

  CPPUNIT_ASSERT( initMsgCollection_->addIfUnique(initMsgFrag2,serializedProds) );
  InitMsgView initMsgView2(&(*serializedProds)[0]);
  CPPUNIT_ASSERT(initMsgView2.outputModuleLabel() == "HLTDEBUG");
  CPPUNIT_ASSERT(initMsgView2.size() == serializedProds->size());

  CPPUNIT_ASSERT(initMsgCollection_->size() == 2);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("hltOutputDQM").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("HLTDEBUG").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("CALIB").get() == 0);

  // *** third INIT message ***

  ref = allocate_frame_with_init_msg("CALIB");
  stor::I2OChain initMsgFrag3(ref);
  CPPUNIT_ASSERT(initMsgFrag3.messageCode() == Header::INIT);

  CPPUNIT_ASSERT( initMsgCollection_->addIfUnique(initMsgFrag3,serializedProds) );
  InitMsgView initMsgView3(&(*serializedProds)[0]);
  CPPUNIT_ASSERT(initMsgView3.outputModuleLabel() == "CALIB");
  CPPUNIT_ASSERT(initMsgView3.size() == serializedProds->size());

  CPPUNIT_ASSERT(initMsgCollection_->size() == 3);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("hltOutputDQM").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("HLTDEBUG").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("CALIB").get() != 0);

  // *** duplicate INIT message ***

  ref = allocate_frame_with_init_msg("CALIB");
  stor::I2OChain initMsgFrag4(ref);
  CPPUNIT_ASSERT(initMsgFrag4.messageCode() == Header::INIT);

  CPPUNIT_ASSERT( ! initMsgCollection_->addIfUnique(initMsgFrag4,serializedProds) );

  CPPUNIT_ASSERT(initMsgCollection_->size() == 3);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("hltOutputDQM").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("HLTDEBUG").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("CALIB").get() != 0);

  // *** cleanup ***

  initMsgCollection_->clear();

  CPPUNIT_ASSERT(initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("hltOutputDQM").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("HLTDEBUG").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModuleLabel("CALIB").get() == 0);
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testInitMsgCollection);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
