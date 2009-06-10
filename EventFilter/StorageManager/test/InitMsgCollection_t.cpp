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

using stor::testhelper::allocate_frame_with_basic_header;
using stor::testhelper::allocate_frame_with_init_msg;

class testInitMsgCollection : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testInitMsgCollection);
  CPPUNIT_TEST(testAdditions);
  CPPUNIT_TEST(testConsumers);

  CPPUNIT_TEST_SUITE_END();

public:
  void testAdditions();
  void testConsumers();

private:
  boost::shared_ptr<InitMsgCollection> _initMsgCollection;
};


void testInitMsgCollection::testAdditions()
{
  if (_initMsgCollection.get() == 0)
    {
      _initMsgCollection.reset(new InitMsgCollection());
    }
  std::vector<unsigned char> tmpBuff;

  CPPUNIT_ASSERT(_initMsgCollection->size() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("hltOutputDQM").get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("HLTDEBUG").get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("CALIB").get() == 0);

  // *** first INIT message ***

  Reference* ref = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  initMsgFrag.copyFragmentsIntoBuffer(tmpBuff);
  InitMsgView view( &tmpBuff[0] );
  _initMsgCollection->addIfUnique(view);

  CPPUNIT_ASSERT(_initMsgCollection->size() == 1);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("hltOutputDQM").get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("HLTDEBUG").get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("CALIB").get() == 0);

  // *** second INIT message ***

  ref = allocate_frame_with_init_msg("HLTDEBUG");
  stor::I2OChain initMsgFrag2(ref);
  CPPUNIT_ASSERT(initMsgFrag2.messageCode() == Header::INIT);

  initMsgFrag2.copyFragmentsIntoBuffer(tmpBuff);
  InitMsgView view2( &tmpBuff[0] );
  _initMsgCollection->addIfUnique(view2);

  CPPUNIT_ASSERT(_initMsgCollection->size() == 2);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("hltOutputDQM").get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("HLTDEBUG").get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("CALIB").get() == 0);

  // *** third INIT message ***

  ref = allocate_frame_with_init_msg("CALIB");
  stor::I2OChain initMsgFrag3(ref);
  CPPUNIT_ASSERT(initMsgFrag3.messageCode() == Header::INIT);

  initMsgFrag3.copyFragmentsIntoBuffer(tmpBuff);
  InitMsgView view3( &tmpBuff[0] );
  _initMsgCollection->addIfUnique(view3);

  CPPUNIT_ASSERT(_initMsgCollection->size() == 3);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("hltOutputDQM").get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("HLTDEBUG").get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("CALIB").get() != 0);

  // *** duplicate INIT message ***

  ref = allocate_frame_with_init_msg("CALIB");
  stor::I2OChain initMsgFrag4(ref);
  CPPUNIT_ASSERT(initMsgFrag4.messageCode() == Header::INIT);

  initMsgFrag4.copyFragmentsIntoBuffer(tmpBuff);
  InitMsgView view4( &tmpBuff[0] );
  _initMsgCollection->addIfUnique(view4);

  CPPUNIT_ASSERT(_initMsgCollection->size() == 3);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("hltOutputDQM").get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("HLTDEBUG").get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("CALIB").get() != 0);

  // *** cleanup ***

  _initMsgCollection->clear();

  CPPUNIT_ASSERT(_initMsgCollection->size() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("hltOutputDQM").get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("HLTDEBUG").get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForOutputModule("CALIB").get() == 0);
}


void testInitMsgCollection::testConsumers()
{
  if (_initMsgCollection.get() == 0)
    {
      _initMsgCollection.reset(new InitMsgCollection());
    }
  std::vector<unsigned char> tmpBuff;

  ConsumerID id0;
  ConsumerID id1(1);
  ConsumerID id2(2);
  ConsumerID id3(3);
  ConsumerID id4(4);

  CPPUNIT_ASSERT(_initMsgCollection->size() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id0).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id1).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id2).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id3).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id4).get() == 0);

  CPPUNIT_ASSERT(! _initMsgCollection->registerConsumer(id0, "hltOutputDQM"));
  CPPUNIT_ASSERT(! _initMsgCollection->registerConsumer(id1, ""));
  CPPUNIT_ASSERT(_initMsgCollection->registerConsumer(id1, "hltOutputDQM"));
  CPPUNIT_ASSERT(_initMsgCollection->registerConsumer(id2, "HLTDEBUG"));
  CPPUNIT_ASSERT(_initMsgCollection->registerConsumer(id3, "HLTDEBUG"));
  CPPUNIT_ASSERT(_initMsgCollection->registerConsumer(id4, "CALIB"));

  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id0).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id1).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id2).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id3).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id4).get() == 0);

  // *** first INIT message ***

  Reference* ref = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  initMsgFrag.copyFragmentsIntoBuffer(tmpBuff);
  InitMsgView view( &tmpBuff[0] );
  _initMsgCollection->addIfUnique(view);

  CPPUNIT_ASSERT(_initMsgCollection->size() == 1);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id0).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id1).get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id2).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id3).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id4).get() == 0);

  // *** second INIT message ***

  ref = allocate_frame_with_init_msg("HLTDEBUG");
  stor::I2OChain initMsgFrag2(ref);
  CPPUNIT_ASSERT(initMsgFrag2.messageCode() == Header::INIT);

  initMsgFrag2.copyFragmentsIntoBuffer(tmpBuff);
  InitMsgView view2( &tmpBuff[0] );
  _initMsgCollection->addIfUnique(view2);

  CPPUNIT_ASSERT(_initMsgCollection->size() == 2);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id0).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id1).get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id2).get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id3).get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id4).get() == 0);

  // *** third INIT message ***

  ref = allocate_frame_with_init_msg("CALIB");
  stor::I2OChain initMsgFrag3(ref);
  CPPUNIT_ASSERT(initMsgFrag3.messageCode() == Header::INIT);

  initMsgFrag3.copyFragmentsIntoBuffer(tmpBuff);
  InitMsgView view3( &tmpBuff[0] );
  _initMsgCollection->addIfUnique(view3);

  CPPUNIT_ASSERT(_initMsgCollection->size() == 3);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id0).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id1).get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id2).get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id3).get() != 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id4).get() != 0);

  // *** cleanup ***

  _initMsgCollection->clear();

  CPPUNIT_ASSERT(_initMsgCollection->size() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id0).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id1).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id2).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id3).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id4).get() == 0);

  // *** new first INIT message ***

  ref = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag4(ref);
  CPPUNIT_ASSERT(initMsgFrag4.messageCode() == Header::INIT);

  initMsgFrag4.copyFragmentsIntoBuffer(tmpBuff);
  InitMsgView view4( &tmpBuff[0] );
  _initMsgCollection->addIfUnique(view4);

  CPPUNIT_ASSERT(_initMsgCollection->size() == 1);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id0).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id1).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id2).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id3).get() == 0);
  CPPUNIT_ASSERT(_initMsgCollection->getElementForConsumer(id4).get() == 0);
}


// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testInitMsgCollection);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
