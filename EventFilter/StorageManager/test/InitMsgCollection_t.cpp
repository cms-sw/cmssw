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
  std::vector<unsigned char> tmpBuff;

  CPPUNIT_ASSERT(initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("hltOutputDQM").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("HLTDEBUG").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("CALIB").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("hltOutputDQM") == 0);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("HLTDEBUG") == 0);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("CALIB") == 0);
  CPPUNIT_ASSERT(initMsgCollection_->maxMsgCount() == 0);

  // *** first INIT message ***

  Reference* ref = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  initMsgFrag.copyFragmentsIntoBuffer(tmpBuff);
  InitMsgView view( &tmpBuff[0] );
  initMsgCollection_->addIfUnique(view);

  CPPUNIT_ASSERT(initMsgCollection_->size() == 1);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("hltOutputDQM").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("HLTDEBUG").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("CALIB").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("hltOutputDQM") == 1);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("HLTDEBUG") == 0);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("CALIB") == 0);
  CPPUNIT_ASSERT(initMsgCollection_->maxMsgCount() == 1);

  // *** second INIT message ***

  ref = allocate_frame_with_init_msg("HLTDEBUG");
  stor::I2OChain initMsgFrag2(ref);
  CPPUNIT_ASSERT(initMsgFrag2.messageCode() == Header::INIT);

  initMsgFrag2.copyFragmentsIntoBuffer(tmpBuff);
  InitMsgView view2( &tmpBuff[0] );
  initMsgCollection_->addIfUnique(view2);

  CPPUNIT_ASSERT(initMsgCollection_->size() == 2);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("hltOutputDQM").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("HLTDEBUG").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("CALIB").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("hltOutputDQM") == 1);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("HLTDEBUG") == 1);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("CALIB") == 0);
  CPPUNIT_ASSERT(initMsgCollection_->maxMsgCount() == 1);

  // *** third INIT message ***

  ref = allocate_frame_with_init_msg("CALIB");
  stor::I2OChain initMsgFrag3(ref);
  CPPUNIT_ASSERT(initMsgFrag3.messageCode() == Header::INIT);

  initMsgFrag3.copyFragmentsIntoBuffer(tmpBuff);
  InitMsgView view3( &tmpBuff[0] );
  initMsgCollection_->addIfUnique(view3);

  CPPUNIT_ASSERT(initMsgCollection_->size() == 3);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("hltOutputDQM").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("HLTDEBUG").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("CALIB").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("hltOutputDQM") == 1);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("HLTDEBUG") == 1);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("CALIB") == 1);
  CPPUNIT_ASSERT(initMsgCollection_->maxMsgCount() == 1);

  // *** duplicate INIT message ***

  ref = allocate_frame_with_init_msg("CALIB");
  stor::I2OChain initMsgFrag4(ref);
  CPPUNIT_ASSERT(initMsgFrag4.messageCode() == Header::INIT);

  initMsgFrag4.copyFragmentsIntoBuffer(tmpBuff);
  InitMsgView view4( &tmpBuff[0] );
  initMsgCollection_->addIfUnique(view4);

  CPPUNIT_ASSERT(initMsgCollection_->size() == 3);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("hltOutputDQM").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("HLTDEBUG").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("CALIB").get() != 0);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("hltOutputDQM") == 1);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("HLTDEBUG") == 1);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("CALIB") == 2);
  CPPUNIT_ASSERT(initMsgCollection_->maxMsgCount() == 2);

  // *** cleanup ***

  initMsgCollection_->clear();

  CPPUNIT_ASSERT(initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("hltOutputDQM").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("HLTDEBUG").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->getElementForOutputModule("CALIB").get() == 0);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("hltOutputDQM") == 0);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("HLTDEBUG") == 0);
  CPPUNIT_ASSERT(initMsgCollection_->initMsgCount("CALIB") == 0);
  CPPUNIT_ASSERT(initMsgCollection_->maxMsgCount() == 0);
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testInitMsgCollection);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
