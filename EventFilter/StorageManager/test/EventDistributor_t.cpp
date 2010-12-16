#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <algorithm>

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/DQMConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"
#include "EventFilter/StorageManager/interface/EventConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"

#include "EventFilter/StorageManager/test/TestHelper.h"
#include "EventFilter/StorageManager/test/MockApplication.h"

#include "DataFormats/Common/interface/HLTenums.h"


/////////////////////////////////////////////////////////////
//
// This test exercises the EventDistributor class
//
/////////////////////////////////////////////////////////////

using namespace stor;

using stor::testhelper::allocate_frame_with_init_msg;
using stor::testhelper::allocate_frame_with_event_msg;
using stor::testhelper::allocate_frame_with_error_msg;
using stor::testhelper::allocate_frame_with_dqm_msg;
using stor::testhelper::set_trigger_bit;
using stor::testhelper::clear_trigger_bits;

class testEventDistributor : public CppUnit::TestFixture
{
  typedef toolbox::mem::Reference Reference;
  CPPUNIT_TEST_SUITE(testEventDistributor);
  CPPUNIT_TEST(testInitMessages);
  CPPUNIT_TEST(testStreamSelection);
  CPPUNIT_TEST(testDuplicatedStreamSelection);
  CPPUNIT_TEST(testConsumerSelection);
  CPPUNIT_TEST(testDuplicatedConsumerSelection);
  CPPUNIT_TEST(testDQMMessages);
  CPPUNIT_TEST(testDuplicatedDQMConsumerSelection);

  CPPUNIT_TEST_SUITE_END();

public:
  void testInitMessages();
  void testStreamSelection();
  void testDuplicatedStreamSelection();
  void testConsumerSelection();
  void testDuplicatedConsumerSelection();
  void testDQMMessages();
  void testDuplicatedDQMConsumerSelection();

private:
  std::string getSampleStreamConfig();
  void initEventDistributor();

  static boost::shared_ptr<SharedResources> _sharedResources;
  static boost::shared_ptr<EventDistributor> _eventDistributor;
};

// 30-Jun-2009, KAB - It turns out that CPPUNIT creates a new instance
// of the test class for each test. So, if we are going to gain
// any value from reusing a single EventDistributor, etc., then we need
// to make those attributes static.  Worse, some XDAQ elements are 
// effectively singletons (such as an infospace with a given name),
// so we need to ensure that only a single instance of the SharedResources
// class gets created.  (Yet another reason to make these static attributes.)
boost::shared_ptr<SharedResources> testEventDistributor::_sharedResources;
boost::shared_ptr<EventDistributor> testEventDistributor::_eventDistributor;

void testEventDistributor::initEventDistributor()
{
  if (_eventDistributor.get() == 0)
    {
      xdaq::Application* app = mockapps::getMockXdaqApplication();
      _sharedResources.reset(new SharedResources());
      _sharedResources->_configuration.reset(new Configuration(app->getApplicationInfoSpace(), 0));
      _sharedResources->_initMsgCollection.reset(new InitMsgCollection());
      _sharedResources->_streamQueue.reset(new StreamQueue(1024));
      _sharedResources->_dqmEventQueue.reset(new DQMEventQueue(1024));
      _sharedResources->_statisticsReporter.reset(new StatisticsReporter(app,_sharedResources));
      _eventDistributor.reset(new EventDistributor(_sharedResources));
      EventConsumerMonitorCollection& ecmc = 
        _sharedResources->_statisticsReporter->getEventConsumerMonitorCollection();
      _sharedResources->_eventConsumerQueueCollection.reset( new EventQueueCollection( ecmc ) );
      DQMConsumerMonitorCollection& dcmc = 
        _sharedResources->_statisticsReporter->getDQMConsumerMonitorCollection();
      _sharedResources->_dqmEventConsumerQueueCollection.reset( new DQMEventQueueCollection( dcmc ) );
    }
  else
  {
    // *** cleanup ***
    
    _sharedResources->_initMsgCollection->clear();
    _eventDistributor->clearStreams();
    _eventDistributor->clearConsumers();
    
    CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 0);
    CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 0);
    CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 0);
    CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 0);
    CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 0);
  }
}


void testEventDistributor::testInitMessages()
{
  initEventDistributor();

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 0);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 0);
  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 0);

  // *** specify configuration ***

  EvtStrConfigListPtr evtCfgList(new EvtStrConfigList);
  ErrStrConfigListPtr errCfgList(new ErrStrConfigList);
  parseStreamConfiguration(getSampleStreamConfig(), evtCfgList, errCfgList);
  _eventDistributor->registerEventStreams(evtCfgList);
  _eventDistributor->registerErrorStreams(errCfgList);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 0);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 0);

  // *** first INIT message ***

  Reference* ref = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  _eventDistributor->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 1);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  // *** second INIT message ***

  ref = allocate_frame_with_init_msg("HLTDEBUG");
  stor::I2OChain initMsgFrag2(ref);
  CPPUNIT_ASSERT(initMsgFrag2.messageCode() == Header::INIT);

  _eventDistributor->addEventToRelevantQueues(initMsgFrag2);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 2);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 2);

  // *** third INIT message ***

  ref = allocate_frame_with_init_msg("CALIB");
  stor::I2OChain initMsgFrag3(ref);
  CPPUNIT_ASSERT(initMsgFrag3.messageCode() == Header::INIT);

  _eventDistributor->addEventToRelevantQueues(initMsgFrag3);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 3);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 3);

  // *** duplicate INIT message ***

  ref = allocate_frame_with_init_msg("CALIB");
  stor::I2OChain initMsgFrag4(ref);
  CPPUNIT_ASSERT(initMsgFrag4.messageCode() == Header::INIT);

  _eventDistributor->addEventToRelevantQueues(initMsgFrag4);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 3);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 3);

  // *** bogus INIT message ***

  ref = allocate_frame_with_init_msg("BOGUS");
  stor::I2OChain initMsgFrag5(ref);
  CPPUNIT_ASSERT(initMsgFrag5.messageCode() == Header::INIT);

  _eventDistributor->addEventToRelevantQueues(initMsgFrag5);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 3);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 4);
  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 0);
}


void testEventDistributor::testStreamSelection()
{
  initEventDistributor();

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 0);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 0);
  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 0);

  // *** specify configuration ***

  EvtStrConfigListPtr evtCfgList(new EvtStrConfigList);
  ErrStrConfigListPtr errCfgList(new ErrStrConfigList);
  parseStreamConfiguration(getSampleStreamConfig(), evtCfgList, errCfgList);
  evtCfgList->at(0).setStreamId(1);
  evtCfgList->at(1).setStreamId(2);
  evtCfgList->at(2).setStreamId(3);
  errCfgList->at(0).setStreamId(4);
  _eventDistributor->registerEventStreams(evtCfgList);
  _eventDistributor->registerErrorStreams(errCfgList);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 0);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 0);

  // *** INIT message ***

  Reference* ref = allocate_frame_with_init_msg("HLTDEBUG");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  _eventDistributor->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 1);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  // *** HLT trigger bit tests ***
  std::vector<unsigned char> hltBits;
  CPPUNIT_ASSERT(hltBits.size() == 0);

  set_trigger_bit(hltBits, 8, edm::hlt::Ready);

  CPPUNIT_ASSERT(hltBits.size() == 3);
  CPPUNIT_ASSERT(hltBits[2] == 0);

  set_trigger_bit(hltBits, 3, edm::hlt::Fail);
  set_trigger_bit(hltBits, 5, edm::hlt::Pass);
  set_trigger_bit(hltBits, 8, edm::hlt::Exception);
  CPPUNIT_ASSERT(hltBits[0] == 0x80);
  CPPUNIT_ASSERT(hltBits[1] == 0x4);
  CPPUNIT_ASSERT(hltBits[2] == 0x3);

  // *** first event message (should pass) ***

  uint32_t eventNumber = 1;
  uint32_t hltBitCount = 9;
  clear_trigger_bits(hltBits);
  set_trigger_bit(hltBits, 0, edm::hlt::Pass);

  ref = allocate_frame_with_event_msg("HLTDEBUG", hltBits, hltBitCount,
                                      eventNumber);
  stor::I2OChain eventMsgFrag(ref);
  CPPUNIT_ASSERT(eventMsgFrag.messageCode() == Header::EVENT);

  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());

  _eventDistributor->addEventToRelevantQueues(eventMsgFrag);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 1);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());

  std::vector<StreamID> streamIdList = eventMsgFrag.getStreamTags();
  CPPUNIT_ASSERT(streamIdList.size() == 1);
  CPPUNIT_ASSERT(std::count(streamIdList.begin(),streamIdList.end(),2) == 1);

  // *** second event message (should not pass) ***

  clear_trigger_bits(hltBits);
  set_trigger_bit(hltBits, 0, edm::hlt::Fail);
  set_trigger_bit(hltBits, 2, edm::hlt::Exception);
  set_trigger_bit(hltBits, 4, edm::hlt::Pass);

  ++eventNumber;
  ref = allocate_frame_with_event_msg("HLTDEBUG", hltBits, hltBitCount,
                                      eventNumber);
  stor::I2OChain eventMsgFrag2(ref);
  CPPUNIT_ASSERT(eventMsgFrag2.messageCode() == Header::EVENT);

  CPPUNIT_ASSERT(!eventMsgFrag2.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag2.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag2.isTaggedForAnyDQMEventConsumer());

  _eventDistributor->addEventToRelevantQueues(eventMsgFrag2);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 1);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  CPPUNIT_ASSERT(!eventMsgFrag2.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag2.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag2.isTaggedForAnyDQMEventConsumer());

  // *** third event message (should not pass) ***

  clear_trigger_bits(hltBits);
  set_trigger_bit(hltBits, 0, edm::hlt::Pass);

  ++eventNumber;
  ref = allocate_frame_with_event_msg("BOGUS", hltBits, hltBitCount,
                                      eventNumber);
  stor::I2OChain eventMsgFrag3(ref);
  CPPUNIT_ASSERT(eventMsgFrag3.messageCode() == Header::EVENT);

  CPPUNIT_ASSERT(!eventMsgFrag3.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag3.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag3.isTaggedForAnyDQMEventConsumer());

  _eventDistributor->addEventToRelevantQueues(eventMsgFrag3);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 1);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  CPPUNIT_ASSERT(!eventMsgFrag3.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag3.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag3.isTaggedForAnyDQMEventConsumer());

  // *** first error message (should pass) ***

  ++eventNumber;
  ref = allocate_frame_with_error_msg(eventNumber);
  stor::I2OChain errorMsgFrag(ref);
  CPPUNIT_ASSERT(errorMsgFrag.messageCode() == Header::ERROR_EVENT);

  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyDQMEventConsumer());

  _eventDistributor->addEventToRelevantQueues(errorMsgFrag);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 1);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  CPPUNIT_ASSERT(errorMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyDQMEventConsumer());

  streamIdList = errorMsgFrag.getStreamTags();
  CPPUNIT_ASSERT(streamIdList.size() == 1);
  CPPUNIT_ASSERT(std::count(streamIdList.begin(),streamIdList.end(),4) == 1);

  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 0);
}


void testEventDistributor::testDuplicatedStreamSelection()
{
  initEventDistributor();

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 0);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 0);
  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 0);

  // *** specify configuration ***

  EvtStrConfigListPtr evtCfgList(new EvtStrConfigList);
  ErrStrConfigListPtr errCfgList(new ErrStrConfigList);
  parseStreamConfiguration(getSampleStreamConfig(), evtCfgList, errCfgList);
  evtCfgList->at(0).setStreamId(1);
  evtCfgList->at(1).setStreamId(2);
  evtCfgList->at(2).setStreamId(3);
  errCfgList->at(0).setStreamId(4);
  _eventDistributor->registerEventStreams(evtCfgList);
  _eventDistributor->registerErrorStreams(errCfgList);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);

  // *** re-register identical event stream ***

  _eventDistributor->registerEventStreams(evtCfgList);
  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);

  // *** re-register identical error event stream ***

  _eventDistributor->registerErrorStreams(errCfgList);
  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);

  // *** register streams with same selection ***

  EvtStrConfigListPtr evtCfgList2(new EvtStrConfigList);
  ErrStrConfigListPtr errCfgList2(new ErrStrConfigList);
  parseStreamConfiguration(getSampleStreamConfig(), evtCfgList2, errCfgList2);
  evtCfgList2->at(0).setStreamId(1);
  evtCfgList2->at(1).setStreamId(2);
  evtCfgList2->at(2).setStreamId(3);
  errCfgList2->at(0).setStreamId(4);
  _eventDistributor->registerEventStreams(evtCfgList2);
  _eventDistributor->registerErrorStreams(errCfgList2);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 4);

  // *** register streams with same selection, but different stream ids *** 

  evtCfgList2->at(1).setStreamId(8);
  _eventDistributor->registerEventStreams(evtCfgList2);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 5);

  // *** INIT message ***

  Reference* ref = allocate_frame_with_init_msg("HLTDEBUG");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  _eventDistributor->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 5);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 2);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);
  
  // *** HLT trigger bit tests ***
  std::vector<unsigned char> hltBits;
  CPPUNIT_ASSERT(hltBits.size() == 0);

  set_trigger_bit(hltBits, 8, edm::hlt::Ready);

  CPPUNIT_ASSERT(hltBits.size() == 3);
  CPPUNIT_ASSERT(hltBits[2] == 0);

  set_trigger_bit(hltBits, 3, edm::hlt::Fail);
  set_trigger_bit(hltBits, 5, edm::hlt::Pass);
  set_trigger_bit(hltBits, 8, edm::hlt::Exception);
  CPPUNIT_ASSERT(hltBits[0] == 0x80);
  CPPUNIT_ASSERT(hltBits[1] == 0x4);
  CPPUNIT_ASSERT(hltBits[2] == 0x3);

  // *** event message (should pass) ***

  uint32_t eventNumber = 1;
  uint32_t hltBitCount = 9;
  clear_trigger_bits(hltBits);
  set_trigger_bit(hltBits, 0, edm::hlt::Pass);

  ref = allocate_frame_with_event_msg("HLTDEBUG", hltBits, hltBitCount,
                                      eventNumber);
  stor::I2OChain eventMsgFrag(ref);
  CPPUNIT_ASSERT(eventMsgFrag.messageCode() == Header::EVENT);

  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());

  _eventDistributor->addEventToRelevantQueues(eventMsgFrag);

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 5);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 2);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());

  std::vector<StreamID> streamIdList = eventMsgFrag.getStreamTags();
  CPPUNIT_ASSERT(streamIdList.size() == 2);
  CPPUNIT_ASSERT(std::count(streamIdList.begin(),streamIdList.end(),2) == 1);
  CPPUNIT_ASSERT(std::count(streamIdList.begin(),streamIdList.end(),8) == 1);
}


void testEventDistributor::testConsumerSelection()
{
  initEventDistributor();

  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 0);
  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 0);

  // *** first consumer ***

  Strings selections;
  std::string sel;
  boost::shared_ptr<EventConsumerRegistrationInfo> consInfo;

  selections.clear();
  selections.push_back("a");
  selections.push_back("b");
  sel = "a || b"; 
  QueueID queueId1(enquing_policy::DiscardOld, 1);
  consInfo.reset(new EventConsumerRegistrationInfo(
      "Test Consumer", sel, selections, "hltOutputDQM",
      queueId1.index(), queueId1.policy(),
      boost::posix_time::seconds(120), "localhost"));
  consInfo->setQueueID( queueId1 );

  _eventDistributor->registerEventConsumer(&(*consInfo));
  
  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 0);

  // *** INIT message ***

  Reference* ref = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  _eventDistributor->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 1);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  // *** second consumer ***
  selections.clear();
  selections.push_back("c");
  selections.push_back("d");
  sel = "c || d"; 
  QueueID queueId2(enquing_policy::DiscardNew, 2);
  consInfo.reset(new EventConsumerRegistrationInfo(
      "Test Consumer", sel, selections, "hltOutputDQM", 
      queueId2.index(), queueId2.policy(),
      boost::posix_time::seconds(120), "localhost" ));
  consInfo->setQueueID( queueId2 );
  
  _eventDistributor->registerEventConsumer(&(*consInfo));
  
  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 2);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 2);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  // *** first event message (should pass both consumers) ***

  std::vector<unsigned char> hltBits;
  set_trigger_bit(hltBits, 8, edm::hlt::Ready);

  uint32_t eventNumber = 1;
  uint32_t hltBitCount = 9;
  clear_trigger_bits(hltBits);
  set_trigger_bit(hltBits, 0, edm::hlt::Pass);
  set_trigger_bit(hltBits, 2, edm::hlt::Pass);

  ref = allocate_frame_with_event_msg("hltOutputDQM", hltBits, hltBitCount,
                                      eventNumber);
  stor::I2OChain eventMsgFrag(ref);
  CPPUNIT_ASSERT(eventMsgFrag.messageCode() == Header::EVENT);

  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());

  _eventDistributor->addEventToRelevantQueues(eventMsgFrag);

  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());

  std::vector<QueueID> queueIdList = eventMsgFrag.getEventConsumerTags();
  CPPUNIT_ASSERT(queueIdList.size() == 2);
  std::vector<QueueID>::const_iterator it =
    std::find(queueIdList.begin(), queueIdList.end(), queueId1);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 1);
  CPPUNIT_ASSERT(it->policy() == enquing_policy::DiscardOld);
  it = std::find(queueIdList.begin(), queueIdList.end(), queueId2);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 2);
  CPPUNIT_ASSERT(it->policy() == enquing_policy::DiscardNew);

  // *** third consumer ***
  selections.clear();
  selections.push_back("c");
  selections.push_back("a");
  sel = "c || a"; 
  QueueID queueId3(enquing_policy::DiscardOld, 3);
  consInfo.reset(new EventConsumerRegistrationInfo(
      "Test Consumer", sel, selections, "hltOutputDQM",
      queueId3.index(), queueId3.policy(),
      boost::posix_time::seconds(120), "localhost"));
  consInfo->setQueueID( queueId3 );
  consInfo->registerMe(&(*_eventDistributor));
    
  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 3);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 3);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  // *** fourth consumer ***
  selections.clear();
  selections.push_back("b");
  selections.push_back("d");
  sel = "b || d"; 
  QueueID queueId4(enquing_policy::DiscardNew, 4);
  consInfo.reset(new EventConsumerRegistrationInfo(
      "Test Consumer", sel, selections, "hltOutputDQM",
      queueId4.index(), queueId4.policy(),
      boost::posix_time::seconds(120), "localhost"));
  consInfo->setQueueID( queueId4 );
  
  consInfo->registerMe(&(*_eventDistributor));
  
  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 4);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);
  
  // *** second event message (should not pass) ***
  
  clear_trigger_bits(hltBits);
  set_trigger_bit(hltBits, 0, edm::hlt::Fail);
  set_trigger_bit(hltBits, 2, edm::hlt::Exception);
  set_trigger_bit(hltBits, 4, edm::hlt::Pass);

  ++eventNumber;
  ref = allocate_frame_with_event_msg("hltOutputDQM", hltBits, hltBitCount,
                                      eventNumber);
  stor::I2OChain eventMsgFrag2(ref);
  CPPUNIT_ASSERT(eventMsgFrag2.messageCode() == Header::EVENT);

  CPPUNIT_ASSERT(!eventMsgFrag2.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag2.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag2.isTaggedForAnyDQMEventConsumer());

  _eventDistributor->addEventToRelevantQueues(eventMsgFrag2);

  CPPUNIT_ASSERT(!eventMsgFrag2.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag2.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag2.isTaggedForAnyDQMEventConsumer());

  // *** third event message (should pass) ***

  clear_trigger_bits(hltBits);
  set_trigger_bit(hltBits, 0, edm::hlt::Pass);
  set_trigger_bit(hltBits, 3, edm::hlt::Pass);
  set_trigger_bit(hltBits, 4, edm::hlt::Pass);
  set_trigger_bit(hltBits, 5, edm::hlt::Pass);
  set_trigger_bit(hltBits, 6, edm::hlt::Pass);
  set_trigger_bit(hltBits, 7, edm::hlt::Pass);

  ++eventNumber;
  ref = allocate_frame_with_event_msg("hltOutputDQM", hltBits, hltBitCount,
                                      eventNumber);
  stor::I2OChain eventMsgFrag3(ref);
  CPPUNIT_ASSERT(eventMsgFrag3.messageCode() == Header::EVENT);

  CPPUNIT_ASSERT(!eventMsgFrag3.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag3.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag3.isTaggedForAnyDQMEventConsumer());

  _eventDistributor->addEventToRelevantQueues(eventMsgFrag3);

  CPPUNIT_ASSERT(!eventMsgFrag3.isTaggedForAnyStream());
  CPPUNIT_ASSERT(eventMsgFrag3.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag3.isTaggedForAnyDQMEventConsumer());

  queueIdList = eventMsgFrag3.getEventConsumerTags();
  CPPUNIT_ASSERT(queueIdList.size() == 4);

  it = std::find(queueIdList.begin(), queueIdList.end(), queueId1);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 1);
  CPPUNIT_ASSERT(it->policy() == enquing_policy::DiscardOld);
  it = std::find(queueIdList.begin(), queueIdList.end(), queueId2);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 2);
  CPPUNIT_ASSERT(it->policy() == enquing_policy::DiscardNew);
  it = std::find(queueIdList.begin(), queueIdList.end(), queueId3);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 3);
  CPPUNIT_ASSERT(it->policy() == enquing_policy::DiscardOld);
  it = std::find(queueIdList.begin(), queueIdList.end(), queueId4);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 4);
  CPPUNIT_ASSERT(it->policy() == enquing_policy::DiscardNew);

  // *** fourth event message (should not pass) ***

  clear_trigger_bits(hltBits);
  set_trigger_bit(hltBits, 0, edm::hlt::Pass);
  set_trigger_bit(hltBits, 1, edm::hlt::Pass);
  set_trigger_bit(hltBits, 2, edm::hlt::Pass);
  set_trigger_bit(hltBits, 3, edm::hlt::Pass);

  ++eventNumber;
  ref = allocate_frame_with_event_msg("BOGUS", hltBits, hltBitCount,
                                      eventNumber);
  stor::I2OChain eventMsgFrag4(ref);
  CPPUNIT_ASSERT(eventMsgFrag4.messageCode() == Header::EVENT);

  CPPUNIT_ASSERT(!eventMsgFrag4.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag4.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag4.isTaggedForAnyDQMEventConsumer());

  _eventDistributor->addEventToRelevantQueues(eventMsgFrag4);

  CPPUNIT_ASSERT(!eventMsgFrag4.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag4.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag4.isTaggedForAnyDQMEventConsumer());

  // *** first error message (should not pass) ***

  ++eventNumber;
  ref = allocate_frame_with_error_msg(eventNumber);
  stor::I2OChain errorMsgFrag(ref);
  CPPUNIT_ASSERT(errorMsgFrag.messageCode() == Header::ERROR_EVENT);

  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyDQMEventConsumer());

  _eventDistributor->addEventToRelevantQueues(errorMsgFrag);

  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyDQMEventConsumer());

  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 4);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 4);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);
  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 0);
}


void testEventDistributor::testDuplicatedConsumerSelection()
{
  initEventDistributor();

  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 0);
  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 0);

  // *** first consumer ***

  Strings selections;
  boost::shared_ptr<EventConsumerRegistrationInfo> consInfo;

  selections.clear();
  selections.push_back("a");
  selections.push_back("b");
  std::string sel = "a || b"; 
  QueueID queueId(enquing_policy::DiscardOld, 1);
  consInfo.reset(new EventConsumerRegistrationInfo(
      "Test Consumer", sel, selections, "hltOutputDQM",
      queueId.index(), queueId.policy(),
      boost::posix_time::seconds(120), "localhost"));
  consInfo->setQueueID( queueId );
  
  _eventDistributor->registerEventConsumer(&(*consInfo));
  
  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 0);

  // *** re-register identical consumer *** //
 
  _eventDistributor->registerEventConsumer(&(*consInfo));
  
  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 0);


  // *** INIT message ***

  Reference* ref = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  _eventDistributor->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 1);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  // *** re-register identical consumer again *** //
 
  _eventDistributor->registerEventConsumer(&(*consInfo));
  
  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 1);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  // *** re-add INIT message *** //

  _eventDistributor->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 1);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);


  // *** second consumer with same selection ***

  boost::shared_ptr<EventConsumerRegistrationInfo> consInfo2;
  consInfo2.reset(new EventConsumerRegistrationInfo(
      "Test Consumer 2", sel, selections, "hltOutputDQM",
      queueId.index(), queueId.policy(),
      boost::posix_time::seconds(120), "remotehost"));
  consInfo2->setQueueID( queueId );
  
  _eventDistributor->registerEventConsumer(&(*consInfo2));
  
  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 1);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  // *** 2nd INIT message ***

  Reference* ref2 = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag2(ref2);
  CPPUNIT_ASSERT(initMsgFrag2.messageCode() == Header::INIT);

  _eventDistributor->addEventToRelevantQueues(initMsgFrag2);

  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 1);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 1);

  // *** event message (should pass) ***

  std::vector<unsigned char> hltBits;
  set_trigger_bit(hltBits, 8, edm::hlt::Ready);

  uint32_t eventNumber = 1;
  uint32_t hltBitCount = 9;
  clear_trigger_bits(hltBits);
  set_trigger_bit(hltBits, 0, edm::hlt::Pass);
  set_trigger_bit(hltBits, 2, edm::hlt::Pass);

  ref = allocate_frame_with_event_msg("hltOutputDQM", hltBits, hltBitCount,
                                      eventNumber);
  stor::I2OChain eventMsgFrag(ref);
  CPPUNIT_ASSERT(eventMsgFrag.messageCode() == Header::EVENT);

  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());

  _eventDistributor->addEventToRelevantQueues(eventMsgFrag);

  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());

  std::vector<QueueID> queueIdList = eventMsgFrag.getEventConsumerTags();
  CPPUNIT_ASSERT(queueIdList.size() == 1);
  CPPUNIT_ASSERT(std::count(queueIdList.begin(),queueIdList.end(),queueId) == 1);
}


std::string testEventDistributor::getSampleStreamConfig()
{
  // should we hard-code this or get it from a file?

  std::stringstream msg;
  msg << "import FWCore.ParameterSet.Config as cms" << std::endl;

  msg << "process = cms.Process(\"SM\")" << std::endl;
  msg << "process.source = cms.Source(\"FragmentInput\")" << std::endl;

  msg << "process.out1 = cms.OutputModule(\"EventStreamFileWriter\"," << std::endl;
  msg << "                                streamLabel = cms.string('A')," << std::endl;
  msg << "                                maxSize = cms.int32(20)," << std::endl;
  msg << "                                SelectHLTOutput = cms.untracked.string('hltOutputDQM')" << std::endl;
  msg << "                                )" << std::endl;

  msg << "process.out2 = cms.OutputModule(\"EventStreamFileWriter\"," << std::endl;
  msg << "                                streamLabel = cms.string('B')," << std::endl;
  msg << "                                maxSize = cms.int32(40)," << std::endl;
  msg << "                                SelectEvents = cms.untracked.PSet( SelectEvents = " << std::endl;
  msg << "                                    cms.vstring('a', 'b', 'c', 'd') )," << std::endl;
  msg << "                                SelectHLTOutput = cms.untracked.string('HLTDEBUG')" << std::endl;
  msg << "                                )" << std::endl;

  msg << "process.out3 = cms.OutputModule(\"EventStreamFileWriter\"," << std::endl;
  msg << "                                streamLabel = cms.string('C')," << std::endl;
  msg << "                                maxSize = cms.int32(60)," << std::endl;
  msg << "                                SelectEvents = cms.untracked.PSet( SelectEvents = " << std::endl;
  msg << "                                    cms.vstring('e') )," << std::endl;
  msg << "                                SelectHLTOutput = cms.untracked.string('CALIB')" << std::endl;
  msg << "                                )" << std::endl;

  msg << "process.out4 = cms.OutputModule(\"ErrorStreamFileWriter\"," << std::endl;
  msg << "                                streamLabel = cms.string('Error')," << std::endl;
  msg << "                                maxSize = cms.int32(1)" << std::endl;
  msg << "                                )" << std::endl;

  msg << "process.end1 = cms.EndPath(process.out1)" << std::endl;
  msg << "process.end2 = cms.EndPath(process.out2)" << std::endl;
  msg << "process.end3 = cms.EndPath(process.out3)" << std::endl;
  msg << "process.end4 = cms.EndPath(process.out4)" << std::endl;

  return msg.str();
}


void testEventDistributor::testDQMMessages()
{

  //
  //// Copied and pasted from methods for other message types: ////
  //
  initEventDistributor();

  //
  //// DQM-specific stuff: ////
  //

  std::string url = "http://localhost:43210/urn:xdaq-application:lid=77";
  enquing_policy::PolicyTag policy = stor::enquing_policy::DiscardOld;

  // Consumer for HCAL:
  boost::shared_ptr<DQMEventConsumerRegistrationInfo> ri1;
  QueueID qid1( policy, 1 );
  ri1.reset( new DQMEventConsumerRegistrationInfo(
      "DQM Consumer 1",
      "HCAL",
      qid1.index(), qid1.policy(),
      boost::posix_time::seconds(10),
      "localhost" )
  );
  ri1->setQueueID( qid1 );
  _eventDistributor->registerDQMEventConsumer( &( *ri1 ) );

  // Consumer for ECAL:
  boost::shared_ptr<DQMEventConsumerRegistrationInfo> ri2;
  QueueID qid2( policy, 2 );
  ri2.reset( new DQMEventConsumerRegistrationInfo(
      "DQM Consumer 2",
      "ECAL",
      qid2.index(), qid2.policy(),
      boost::posix_time::seconds(10), "localhost" )
  );
  ri2->setQueueID( qid2 );
  _eventDistributor->registerDQMEventConsumer( &( *ri2 ) );

  // HCAL event:
  Reference* ref1 = allocate_frame_with_dqm_msg( 1111, "HCAL" );
  stor::I2OChain frag1( ref1 );
  CPPUNIT_ASSERT( frag1.messageCode() == Header::DQM_EVENT );
  _eventDistributor->addEventToRelevantQueues( frag1 );
  CPPUNIT_ASSERT( frag1.isTaggedForAnyDQMEventConsumer() );
  std::vector<QueueID> queueIdList1 = frag1.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList1.size() == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList1.begin(),queueIdList1.end(),qid1) == 1 );

  // ECAL event:
  Reference* ref2 = allocate_frame_with_dqm_msg( 2222, "ECAL" );
  stor::I2OChain frag2( ref2 );
  CPPUNIT_ASSERT( frag2.messageCode() == Header::DQM_EVENT );
  _eventDistributor->addEventToRelevantQueues( frag2 );
  CPPUNIT_ASSERT( frag2.isTaggedForAnyDQMEventConsumer() );
  std::vector<QueueID> queueIdList2 = frag2.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList2.size() == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList2.begin(),queueIdList2.end(),qid2) == 1 );

  // GT event:
  Reference* ref3 = allocate_frame_with_dqm_msg( 3333, "GT" );
  stor::I2OChain frag3( ref3 );
  CPPUNIT_ASSERT( frag3.messageCode() == Header::DQM_EVENT );
  _eventDistributor->addEventToRelevantQueues( frag3 );
  CPPUNIT_ASSERT( !frag3.isTaggedForAnyDQMEventConsumer() );
  CPPUNIT_ASSERT( frag3.getDQMEventConsumerTags().size() == 0 );

  // Wildcard consumer:
  boost::shared_ptr<DQMEventConsumerRegistrationInfo> ri3;
  QueueID qid3( policy, 3 );
  ri3.reset( new DQMEventConsumerRegistrationInfo(
      "DQM Consumer 3",
      "*",
      qid3.index(), qid3.policy(),
      boost::posix_time::seconds(10), "localhost" )
  );
  ri3->setQueueID( qid3 );
  _eventDistributor->registerDQMEventConsumer( &( *ri3 ) );

  // Another HCAL event:
  Reference* ref4 = allocate_frame_with_dqm_msg( 4444, "HCAL" );
  stor::I2OChain frag4( ref4 );
  CPPUNIT_ASSERT( frag4.messageCode() == Header::DQM_EVENT );
  _eventDistributor->addEventToRelevantQueues( frag4 );
  CPPUNIT_ASSERT( frag4.isTaggedForAnyDQMEventConsumer() );
  std::vector<QueueID> queueIdList4 = frag4.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList4.size() == 2 );
  CPPUNIT_ASSERT( std::count(queueIdList4.begin(),queueIdList4.end(),qid1) == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList4.begin(),queueIdList4.end(),qid3) == 1 );

  // Another ECAL event:
  Reference* ref5 = allocate_frame_with_dqm_msg( 5555, "ECAL" );
  stor::I2OChain frag5( ref5 );
  CPPUNIT_ASSERT( frag5.messageCode() == Header::DQM_EVENT );
  _eventDistributor->addEventToRelevantQueues( frag5 );
  CPPUNIT_ASSERT( frag5.isTaggedForAnyDQMEventConsumer() );
  std::vector<QueueID> queueIdList5 = frag5.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList5.size() == 2 );
  CPPUNIT_ASSERT( std::count(queueIdList5.begin(),queueIdList5.end(),qid2) == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList5.begin(),queueIdList5.end(),qid3) == 1 );

  // Another GT event:
  Reference* ref6 = allocate_frame_with_dqm_msg( 6666, "GT" );
  stor::I2OChain frag6( ref6 );
  CPPUNIT_ASSERT( frag6.messageCode() == Header::DQM_EVENT );
  _eventDistributor->addEventToRelevantQueues( frag6 );
  CPPUNIT_ASSERT( frag6.isTaggedForAnyDQMEventConsumer() );
  std::vector<QueueID> queueIdList6 = frag6.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList6.size() == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList6.begin(),queueIdList6.end(),qid3) == 1 );
}


void testEventDistributor::testDuplicatedDQMConsumerSelection()
{
  initEventDistributor();

  CPPUNIT_ASSERT(_eventDistributor->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedStreamCount() == 0);
  CPPUNIT_ASSERT(_sharedResources->_initMsgCollection->size() == 0);
  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(_eventDistributor->initializedConsumerCount() == 0);

  std::string url = "http://localhost:43210/urn:xdaq-application:lid=77";
  enquing_policy::PolicyTag policy = stor::enquing_policy::DiscardOld;

  // Consumer for HCAL:
  boost::shared_ptr<DQMEventConsumerRegistrationInfo> ri1;
  QueueID qid1( policy, 1 );
  ri1.reset( new DQMEventConsumerRegistrationInfo(
      "DQM Consumer 1",
      "HCAL",
      qid1.index(), qid1.policy(),
      boost::posix_time::seconds(10),
      "localhost" )
  );
  ri1->setQueueID( qid1 );
  _eventDistributor->registerDQMEventConsumer( &( *ri1 ) );

  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 1);

  // re-register same consumer

  _eventDistributor->registerDQMEventConsumer( &( *ri1 ) );

  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 1);

  // 2nd consumer requesting same events
  boost::shared_ptr<DQMEventConsumerRegistrationInfo> ri2;
  ri2.reset( new DQMEventConsumerRegistrationInfo(
      "DQM Consumer 2",
      "HCAL",
      qid1.index(), qid1.policy(),
      boost::posix_time::seconds(10),
      "remotehost" )
  );
  ri2->setQueueID( qid1 );
  _eventDistributor->registerDQMEventConsumer( &( *ri2 ) );

  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 1);

  // HCAL event:
  Reference* ref1 = allocate_frame_with_dqm_msg( 1111, "HCAL" );
  stor::I2OChain frag1( ref1 );
  CPPUNIT_ASSERT( frag1.messageCode() == Header::DQM_EVENT );
  _eventDistributor->addEventToRelevantQueues( frag1 );
  CPPUNIT_ASSERT( frag1.isTaggedForAnyDQMEventConsumer() );
  std::vector<QueueID> queueIdList1 = frag1.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList1.size() == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList1.begin(),queueIdList1.end(),qid1) == 1 );

  // 3rd consumer requesting same events, but different queue
  boost::shared_ptr<DQMEventConsumerRegistrationInfo> ri3;
  QueueID qid2( policy, 10 );
  ri3.reset( new DQMEventConsumerRegistrationInfo(
      "DQM Consumer 3",
      "HCAL",
      qid2.index(), qid2.policy(),
      boost::posix_time::seconds(10),
      "remotehost" )
  );
  ri3->setQueueID( qid2 );
  _eventDistributor->registerDQMEventConsumer( &( *ri3 ) );

  CPPUNIT_ASSERT(_eventDistributor->configuredConsumerCount() == 2);

  // Another HCAL event:
  Reference* ref2 = allocate_frame_with_dqm_msg( 4444, "HCAL" );
  stor::I2OChain frag2( ref2 );
  CPPUNIT_ASSERT( frag2.messageCode() == Header::DQM_EVENT );
  _eventDistributor->addEventToRelevantQueues( frag2 );
  CPPUNIT_ASSERT( frag2.isTaggedForAnyDQMEventConsumer() );
  std::vector<QueueID> queueIdList2 = frag2.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList2.size() == 2 );
  CPPUNIT_ASSERT( std::count(queueIdList2.begin(),queueIdList2.end(),qid1) == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList2.begin(),queueIdList2.end(),qid2) == 1 );
}



// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testEventDistributor);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
