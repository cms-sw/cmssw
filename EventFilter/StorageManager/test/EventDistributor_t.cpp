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
  CPPUNIT_TEST(testSharedConsumerSelection);
  CPPUNIT_TEST(testPrescaledConsumerSelection);
  CPPUNIT_TEST(testDQMMessages);
  CPPUNIT_TEST(testDuplicatedDQMConsumerSelection);

  CPPUNIT_TEST_SUITE_END();

public:
  void testInitMessages();
  void testStreamSelection();
  void testDuplicatedStreamSelection();
  void testConsumerSelection();
  void testDuplicatedConsumerSelection();
  void testSharedConsumerSelection();
  void testPrescaledConsumerSelection();
  void testDQMMessages();
  void testDuplicatedDQMConsumerSelection();

private:
  std::string getSampleStreamConfig();
  void initEventDistributor();

  static boost::shared_ptr<SharedResources> sharedResources_;
  static boost::shared_ptr<EventDistributor> eventDistributor_;
};

// 30-Jun-2009, KAB - It turns out that CPPUNIT creates a new instance
// of the test class for each test. So, if we are going to gain
// any value from reusing a single EventDistributor, etc., then we need
// to make those attributes static.  Worse, some XDAQ elements are 
// effectively singletons (such as an infospace with a given name),
// so we need to ensure that only a single instance of the SharedResources
// class gets created.  (Yet another reason to make these static attributes.)
boost::shared_ptr<SharedResources> testEventDistributor::sharedResources_;
boost::shared_ptr<EventDistributor> testEventDistributor::eventDistributor_;

void testEventDistributor::initEventDistributor()
{
  if (eventDistributor_.get() == 0)
    {
      xdaq::Application* app = mockapps::getMockXdaqApplication();
      sharedResources_.reset(new SharedResources());
      sharedResources_->configuration_.reset(new Configuration(app->getApplicationInfoSpace(), 0));
      sharedResources_->initMsgCollection_.reset(new InitMsgCollection());
      sharedResources_->streamQueue_.reset(new StreamQueue(1024));
      sharedResources_->dqmEventQueue_.reset(new DQMEventQueue(1024));
      sharedResources_->statisticsReporter_.reset(new StatisticsReporter(app,sharedResources_));
      eventDistributor_.reset(new EventDistributor(sharedResources_));
      EventConsumerMonitorCollection& ecmc = 
        sharedResources_->statisticsReporter_->getEventConsumerMonitorCollection();
      sharedResources_->eventQueueCollection_.reset( new EventQueueCollection( ecmc ) );
      DQMConsumerMonitorCollection& dcmc = 
        sharedResources_->statisticsReporter_->getDQMConsumerMonitorCollection();
      sharedResources_->dqmEventQueueCollection_.reset( new DQMEventQueueCollection( dcmc ) );
    }
  else
  {
    // *** cleanup ***
    
    sharedResources_->initMsgCollection_->clear();
    eventDistributor_->clearStreams();
    eventDistributor_->clearConsumers();
    sharedResources_->eventQueueCollection_->removeQueues();
    sharedResources_->dqmEventQueueCollection_->removeQueues();

    CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 0);
    CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 0);
    CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
    CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 0);
    CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
    CPPUNIT_ASSERT(sharedResources_->eventQueueCollection_->size() == 0);
    CPPUNIT_ASSERT(sharedResources_->dqmEventQueueCollection_->size() == 0);
  }
}


void testEventDistributor::testInitMessages()
{
  initEventDistributor();

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);

  // *** specify configuration ***

  EvtStrConfigListPtr evtCfgList(new EvtStrConfigList);
  ErrStrConfigListPtr errCfgList(new ErrStrConfigList);
  parseStreamConfiguration(getSampleStreamConfig(), evtCfgList, errCfgList);
  eventDistributor_->registerEventStreams(evtCfgList);
  eventDistributor_->registerErrorStreams(errCfgList);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);

  // *** first INIT message ***

  Reference* ref = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  eventDistributor_->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 1);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

  // *** second INIT message ***

  ref = allocate_frame_with_init_msg("HLTDEBUG");
  stor::I2OChain initMsgFrag2(ref);
  CPPUNIT_ASSERT(initMsgFrag2.messageCode() == Header::INIT);

  eventDistributor_->addEventToRelevantQueues(initMsgFrag2);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 2);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 2);

  // *** third INIT message ***

  ref = allocate_frame_with_init_msg("CALIB");
  stor::I2OChain initMsgFrag3(ref);
  CPPUNIT_ASSERT(initMsgFrag3.messageCode() == Header::INIT);

  eventDistributor_->addEventToRelevantQueues(initMsgFrag3);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 3);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 3);

  // *** duplicate INIT message ***

  ref = allocate_frame_with_init_msg("CALIB");
  stor::I2OChain initMsgFrag4(ref);
  CPPUNIT_ASSERT(initMsgFrag4.messageCode() == Header::INIT);

  eventDistributor_->addEventToRelevantQueues(initMsgFrag4);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 3);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 3);

  // *** bogus INIT message ***

  ref = allocate_frame_with_init_msg("BOGUS");
  stor::I2OChain initMsgFrag5(ref);
  CPPUNIT_ASSERT(initMsgFrag5.messageCode() == Header::INIT);

  eventDistributor_->addEventToRelevantQueues(initMsgFrag5);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 3);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 4);
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
}


void testEventDistributor::testStreamSelection()
{
  initEventDistributor();

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);

  // *** specify configuration ***

  EvtStrConfigListPtr evtCfgList(new EvtStrConfigList);
  ErrStrConfigListPtr errCfgList(new ErrStrConfigList);
  parseStreamConfiguration(getSampleStreamConfig(), evtCfgList, errCfgList);
  evtCfgList->at(0).setStreamId(1);
  evtCfgList->at(1).setStreamId(2);
  evtCfgList->at(2).setStreamId(3);
  errCfgList->at(0).setStreamId(4);
  eventDistributor_->registerEventStreams(evtCfgList);
  eventDistributor_->registerErrorStreams(errCfgList);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);

  // *** INIT message ***

  Reference* ref = allocate_frame_with_init_msg("HLTDEBUG");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  eventDistributor_->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 1);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

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

  eventDistributor_->addEventToRelevantQueues(eventMsgFrag);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 1);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

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

  eventDistributor_->addEventToRelevantQueues(eventMsgFrag2);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 1);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

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

  eventDistributor_->addEventToRelevantQueues(eventMsgFrag3);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 1);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

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

  eventDistributor_->addEventToRelevantQueues(errorMsgFrag);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 1);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

  CPPUNIT_ASSERT(errorMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyDQMEventConsumer());

  streamIdList = errorMsgFrag.getStreamTags();
  CPPUNIT_ASSERT(streamIdList.size() == 1);
  CPPUNIT_ASSERT(std::count(streamIdList.begin(),streamIdList.end(),4) == 1);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
}


void testEventDistributor::testDuplicatedStreamSelection()
{
  initEventDistributor();

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);

  // *** specify configuration ***

  EvtStrConfigListPtr evtCfgList(new EvtStrConfigList);
  ErrStrConfigListPtr errCfgList(new ErrStrConfigList);
  parseStreamConfiguration(getSampleStreamConfig(), evtCfgList, errCfgList);
  evtCfgList->at(0).setStreamId(1);
  evtCfgList->at(1).setStreamId(2);
  evtCfgList->at(2).setStreamId(3);
  errCfgList->at(0).setStreamId(4);
  eventDistributor_->registerEventStreams(evtCfgList);
  eventDistributor_->registerErrorStreams(errCfgList);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);

  // *** re-register identical event stream ***

  eventDistributor_->registerEventStreams(evtCfgList);
  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);

  // *** re-register identical error event stream ***

  eventDistributor_->registerErrorStreams(errCfgList);
  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);

  // *** register streams with same selection ***

  EvtStrConfigListPtr evtCfgList2(new EvtStrConfigList);
  ErrStrConfigListPtr errCfgList2(new ErrStrConfigList);
  parseStreamConfiguration(getSampleStreamConfig(), evtCfgList2, errCfgList2);
  evtCfgList2->at(0).setStreamId(1);
  evtCfgList2->at(1).setStreamId(2);
  evtCfgList2->at(2).setStreamId(3);
  errCfgList2->at(0).setStreamId(4);
  eventDistributor_->registerEventStreams(evtCfgList2);
  eventDistributor_->registerErrorStreams(errCfgList2);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 4);

  // *** register streams with same selection, but different stream ids *** 

  evtCfgList2->at(1).setStreamId(8);
  eventDistributor_->registerEventStreams(evtCfgList2);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 5);

  // *** INIT message ***

  Reference* ref = allocate_frame_with_init_msg("HLTDEBUG");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  eventDistributor_->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 5);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 2);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);
  
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

  eventDistributor_->addEventToRelevantQueues(eventMsgFrag);

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 5);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 2);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

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

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 0);

  boost::shared_ptr<EventConsumerRegistrationInfo> consInfo;
  ConsumerID cid;

  // *** first consumer ***
  edm::ParameterSet pset;
  pset.addUntrackedParameter<std::string>("consumerName", "Test Consumer");
  pset.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  pset.addUntrackedParameter<std::string>("TriggerSelector", "a || b");
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardOld");

  consInfo.reset(new EventConsumerRegistrationInfo(pset));
  consInfo->setConsumerId(++cid);
  QueueID queueId1 = sharedResources_->eventQueueCollection_->createQueue(consInfo);
  CPPUNIT_ASSERT(queueId1.isValid());
  consInfo->setQueueId(queueId1);

  eventDistributor_->registerEventConsumer(consInfo);
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(sharedResources_->eventQueueCollection_->size() == 1);

  // *** INIT message ***

  Reference* ref = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  eventDistributor_->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 1);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

  // *** second consumer ***
  pset.addUntrackedParameter<std::string>("TriggerSelector", "c || d");
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardNew");
  consInfo.reset(new EventConsumerRegistrationInfo(pset));
  consInfo->setConsumerId(++cid);
  QueueID queueId2 = sharedResources_->eventQueueCollection_->createQueue(consInfo);
  CPPUNIT_ASSERT(queueId2.isValid());
  consInfo->setQueueId(queueId2);
  
  eventDistributor_->registerEventConsumer(consInfo);
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 2);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 2);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);
  CPPUNIT_ASSERT(sharedResources_->eventQueueCollection_->size() == 2);

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

  eventDistributor_->addEventToRelevantQueues(eventMsgFrag);

  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());

  QueueIDs queueIdList = eventMsgFrag.getEventConsumerTags();
  CPPUNIT_ASSERT(queueIdList.size() == 2);
  QueueIDs::const_iterator it =
    std::find(queueIdList.begin(), queueIdList.end(), queueId1);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 0);
  CPPUNIT_ASSERT(it->policy() == enquing_policy::DiscardOld);
  it = std::find(queueIdList.begin(), queueIdList.end(), queueId2);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 0);
  CPPUNIT_ASSERT(it->policy() == enquing_policy::DiscardNew);

  // *** third consumer ***
  pset.addUntrackedParameter<std::string>("TriggerSelector", "c || a");
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardOld");
  consInfo.reset(new EventConsumerRegistrationInfo(pset));
  consInfo->setConsumerId(++cid);
  QueueID queueId3 = sharedResources_->eventQueueCollection_->createQueue(consInfo);
  CPPUNIT_ASSERT(queueId3.isValid());
    consInfo->setQueueId(queueId3);
  consInfo->registerMe(&(*eventDistributor_));
    
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 3);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 3);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);
  CPPUNIT_ASSERT(sharedResources_->eventQueueCollection_->size() == 3);

  // *** fourth consumer ***
  pset.addUntrackedParameter<std::string>("TriggerSelector", "b || d");
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardNew");
  consInfo.reset(new EventConsumerRegistrationInfo(pset));
  consInfo->setConsumerId(++cid);
  QueueID queueId4 = sharedResources_->eventQueueCollection_->createQueue(consInfo);
  CPPUNIT_ASSERT(queueId4.isValid());
  consInfo->setQueueId(queueId4);
  
  consInfo->registerMe(&(*eventDistributor_));
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 4);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);
  CPPUNIT_ASSERT(sharedResources_->eventQueueCollection_->size() == 4);
  
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

  eventDistributor_->addEventToRelevantQueues(eventMsgFrag2);

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

  eventDistributor_->addEventToRelevantQueues(eventMsgFrag3);

  CPPUNIT_ASSERT(!eventMsgFrag3.isTaggedForAnyStream());
  CPPUNIT_ASSERT(eventMsgFrag3.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag3.isTaggedForAnyDQMEventConsumer());

  queueIdList = eventMsgFrag3.getEventConsumerTags();
  CPPUNIT_ASSERT(queueIdList.size() == 4);

  it = std::find(queueIdList.begin(), queueIdList.end(), queueId1);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 0);
  CPPUNIT_ASSERT(it->policy() == enquing_policy::DiscardOld);
  it = std::find(queueIdList.begin(), queueIdList.end(), queueId2);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 0);
  CPPUNIT_ASSERT(it->policy() == enquing_policy::DiscardNew);
  it = std::find(queueIdList.begin(), queueIdList.end(), queueId3);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 1);
  CPPUNIT_ASSERT(it->policy() == enquing_policy::DiscardOld);
  it = std::find(queueIdList.begin(), queueIdList.end(), queueId4);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 1);
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

  eventDistributor_->addEventToRelevantQueues(eventMsgFrag4);

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

  eventDistributor_->addEventToRelevantQueues(errorMsgFrag);

  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!errorMsgFrag.isTaggedForAnyDQMEventConsumer());

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 4);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 4);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);
  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 0);
}


void testEventDistributor::testDuplicatedConsumerSelection()
{
  initEventDistributor();

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 0);

  // *** first consumer ***

  Strings selections;
  boost::shared_ptr<EventConsumerRegistrationInfo> consInfo;
  ConsumerID cid;

  selections.clear();
  selections.push_back("a");
  selections.push_back("b");
  edm::ParameterSet pset;
  pset.addUntrackedParameter<std::string>("consumerName", "Test Consumer");
  pset.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  pset.addParameter<Strings>("TrackedEventSelection", selections);
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardOld");
  
  consInfo.reset(new EventConsumerRegistrationInfo(pset));
  consInfo->setConsumerId(++cid);
  QueueID queueId = sharedResources_->eventQueueCollection_->createQueue(consInfo);
  CPPUNIT_ASSERT(queueId.isValid());
  consInfo->setQueueId(queueId);
  
  eventDistributor_->registerEventConsumer(consInfo);
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(sharedResources_->eventQueueCollection_->size() == 1);

  // *** re-register identical consumer *** //
 
  eventDistributor_->registerEventConsumer(consInfo);
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(sharedResources_->eventQueueCollection_->size() == 1);


  // *** INIT message ***

  Reference* ref = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  eventDistributor_->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 1);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

  // *** re-register identical consumer again *** //
 
  eventDistributor_->registerEventConsumer(consInfo);
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 1);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

  // *** re-add INIT message *** //

  eventDistributor_->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 1);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);


  // *** second consumer with same selection ***

  boost::shared_ptr<EventConsumerRegistrationInfo> consInfo2;
  pset.addUntrackedParameter<std::string>("consumerName", "Test Consumer 2");

  consInfo2.reset(new EventConsumerRegistrationInfo(pset, "remotehost"));
  consInfo2->setConsumerId(++cid);
  QueueID queueId2 = sharedResources_->eventQueueCollection_->createQueue(consInfo2);
  CPPUNIT_ASSERT(queueId2.isValid());
  consInfo2->setQueueId(queueId2);
  
  eventDistributor_->registerEventConsumer(consInfo2);
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 2);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 2);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

  // *** 2nd INIT message ***

  Reference* ref2 = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag2(ref2);
  CPPUNIT_ASSERT(initMsgFrag2.messageCode() == Header::INIT);

  eventDistributor_->addEventToRelevantQueues(initMsgFrag2);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 2);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 2);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

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

  eventDistributor_->addEventToRelevantQueues(eventMsgFrag);

  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());

  QueueIDs queueIdList = eventMsgFrag.getEventConsumerTags();
  CPPUNIT_ASSERT(queueIdList.size() == 2);
  CPPUNIT_ASSERT(std::count(queueIdList.begin(),queueIdList.end(),queueId) == 1);
  CPPUNIT_ASSERT(std::count(queueIdList.begin(),queueIdList.end(),queueId2) == 1);
}


void testEventDistributor::testSharedConsumerSelection()
{
  initEventDistributor();
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 0);

  boost::shared_ptr<EventConsumerRegistrationInfo> consInfo;
  ConsumerID cid;

  // *** first consumer ***
  edm::ParameterSet pset;
  pset.addUntrackedParameter<std::string>("consumerName", "Test Consumer 1");
  pset.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  pset.addUntrackedParameter<std::string>("TriggerSelector", "a || b");
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardOld");
  pset.addUntrackedParameter<bool>("uniqueEvents", true);
  consInfo.reset(new EventConsumerRegistrationInfo(pset, "localhost"));
  consInfo->setConsumerId(++cid);
  QueueID qid1 = sharedResources_->eventQueueCollection_->createQueue(consInfo);
  CPPUNIT_ASSERT(qid1.isValid());
  consInfo->setQueueId(qid1);
  
  eventDistributor_->registerEventConsumer(consInfo);
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(sharedResources_->eventQueueCollection_->size() == 1);

  // *** second consumer - same request, but no unique events ***
  pset.addUntrackedParameter<std::string>("consumerName", "Test Consumer 2");
  pset.addUntrackedParameter<bool>("uniqueEvents", false);
  consInfo.reset(new EventConsumerRegistrationInfo(pset, "remotehost"));
  consInfo->setConsumerId(++cid);
  QueueID qid2 = sharedResources_->eventQueueCollection_->createQueue(consInfo);
  CPPUNIT_ASSERT(qid2.isValid());
  consInfo->setQueueId(qid2);
  CPPUNIT_ASSERT(qid1 != qid2);

  eventDistributor_->registerEventConsumer(consInfo);
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 2);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(sharedResources_->eventQueueCollection_->size() == 2);

  // *** third consumer - share with consumer 1 ***
  pset.addUntrackedParameter<std::string>("consumerName", "Test Consumer 3");
  pset.addUntrackedParameter<bool>("uniqueEvents", true);
  consInfo.reset(new EventConsumerRegistrationInfo(pset, "farawayhost"));
  consInfo->setConsumerId(++cid);
  QueueID qid3 = sharedResources_->eventQueueCollection_->createQueue(consInfo);
  CPPUNIT_ASSERT(qid3.isValid());
  consInfo->setQueueId(qid3);
  CPPUNIT_ASSERT(qid1 == qid3);
  CPPUNIT_ASSERT(qid2 != qid3);
  
  eventDistributor_->registerEventConsumer(consInfo);
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 2);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(sharedResources_->eventQueueCollection_->size() == 2);

  // *** INIT message ***

  Reference* ref = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  eventDistributor_->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 2);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 2);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

  // *** event message ***

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

  eventDistributor_->addEventToRelevantQueues(eventMsgFrag);

  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
  CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyEventConsumer());
  CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());

  QueueIDs queueIdList = eventMsgFrag.getEventConsumerTags();
  CPPUNIT_ASSERT(queueIdList.size() == 2);
  QueueIDs::const_iterator it =
    std::find(queueIdList.begin(), queueIdList.end(), qid1);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 0);
  CPPUNIT_ASSERT(it->policy() == enquing_policy::DiscardOld);
  it = std::find(queueIdList.begin(), queueIdList.end(), qid2);
  CPPUNIT_ASSERT(it != queueIdList.end());
  CPPUNIT_ASSERT(it->index() == 1);
  CPPUNIT_ASSERT(it->policy() == enquing_policy::DiscardOld);

}


void testEventDistributor::testPrescaledConsumerSelection()
{
  initEventDistributor();
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 0);

  boost::shared_ptr<EventConsumerRegistrationInfo> consInfo;
  ConsumerID cid;

  // *** first consumer - unprescaled ***
  edm::ParameterSet pset;
  pset.addUntrackedParameter<std::string>("consumerName", "Test Consumer 1");
  pset.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  pset.addUntrackedParameter<std::string>("TriggerSelector", "a || b");
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardOld");
  consInfo.reset(new EventConsumerRegistrationInfo(pset));
  consInfo->setConsumerId(++cid);
  QueueID qid1 = sharedResources_->eventQueueCollection_->createQueue(consInfo);
  CPPUNIT_ASSERT(qid1.isValid());
  consInfo->setQueueId(qid1);
  
  eventDistributor_->registerEventConsumer(consInfo);
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(sharedResources_->eventQueueCollection_->size() == 1);

  // *** second consumer - prescaled by 3 ***
  pset.addUntrackedParameter<int>("prescale", 3);
  consInfo.reset(new EventConsumerRegistrationInfo(pset));
  consInfo->setConsumerId(++cid);
  QueueID qid2 = sharedResources_->eventQueueCollection_->createQueue(consInfo);
  CPPUNIT_ASSERT(qid2.isValid());
  consInfo->setQueueId(qid2);
  
  eventDistributor_->registerEventConsumer(consInfo);
  
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 2);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(sharedResources_->eventQueueCollection_->size() == 2);

  // *** INIT message ***

  Reference* ref = allocate_frame_with_init_msg("hltOutputDQM");
  stor::I2OChain initMsgFrag(ref);
  CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

  eventDistributor_->addEventToRelevantQueues(initMsgFrag);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 2);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 2);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 1);

  // *** first event message - does not pass prescale ***

  std::vector<unsigned char> hltBits;
  set_trigger_bit(hltBits, 8, edm::hlt::Ready);

  uint32_t eventNumber = 1;
  uint32_t hltBitCount = 9;
  clear_trigger_bits(hltBits);
  set_trigger_bit(hltBits, 0, edm::hlt::Pass);
  set_trigger_bit(hltBits, 2, edm::hlt::Pass);

  {
    ref = allocate_frame_with_event_msg("hltOutputDQM", hltBits, hltBitCount,
      eventNumber);
    stor::I2OChain eventMsgFrag(ref);
    CPPUNIT_ASSERT(eventMsgFrag.messageCode() == Header::EVENT);
    
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyEventConsumer());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());
    
    eventDistributor_->addEventToRelevantQueues(eventMsgFrag);
    
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
    CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyEventConsumer());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());
    
    QueueIDs queueIdList = eventMsgFrag.getEventConsumerTags();
    CPPUNIT_ASSERT(queueIdList.size() == 1);
    QueueIDs::const_iterator it =
      std::find(queueIdList.begin(), queueIdList.end(), qid1);
    CPPUNIT_ASSERT(it != queueIdList.end());
    CPPUNIT_ASSERT(it->index() == 0);
  }

  // *** second event message - does not pass prescale ***
  {
    eventNumber = 3;
    
    ref = allocate_frame_with_event_msg("hltOutputDQM", hltBits, hltBitCount,
      eventNumber);
    stor::I2OChain eventMsgFrag(ref);
    CPPUNIT_ASSERT(eventMsgFrag.messageCode() == Header::EVENT);
    
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyEventConsumer());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());
    
    eventDistributor_->addEventToRelevantQueues(eventMsgFrag);
    
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
    CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyEventConsumer());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());
    
    QueueIDs queueIdList = eventMsgFrag.getEventConsumerTags();
    CPPUNIT_ASSERT(queueIdList.size() == 1);
    QueueIDs::const_iterator it =
      std::find(queueIdList.begin(), queueIdList.end(), qid1);
    CPPUNIT_ASSERT(it != queueIdList.end());
    CPPUNIT_ASSERT(it->index() == 0);
  }

  // *** third event message - does pass prescale ***
  {
    eventNumber = 5;
    
    ref = allocate_frame_with_event_msg("hltOutputDQM", hltBits, hltBitCount,
      eventNumber);
    stor::I2OChain eventMsgFrag(ref);
    CPPUNIT_ASSERT(eventMsgFrag.messageCode() == Header::EVENT);
    
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyEventConsumer());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());
    
    eventDistributor_->addEventToRelevantQueues(eventMsgFrag);
    
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
    CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyEventConsumer());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());
    
    QueueIDs queueIdList = eventMsgFrag.getEventConsumerTags();
    CPPUNIT_ASSERT(queueIdList.size() == 2);
    QueueIDs::const_iterator it =
      std::find(queueIdList.begin(), queueIdList.end(), qid1);
    CPPUNIT_ASSERT(it != queueIdList.end());
    CPPUNIT_ASSERT(it->index() == 0);
    it = std::find(queueIdList.begin(), queueIdList.end(), qid2);
    CPPUNIT_ASSERT(it != queueIdList.end());
    CPPUNIT_ASSERT(it->index() == 1);
  }

  // *** forth event message - does not pass prescale ***
  {
    eventNumber = 9;
    
    ref = allocate_frame_with_event_msg("hltOutputDQM", hltBits, hltBitCount,
      eventNumber);
    stor::I2OChain eventMsgFrag(ref);
    CPPUNIT_ASSERT(eventMsgFrag.messageCode() == Header::EVENT);
    
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyEventConsumer());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());
    
    eventDistributor_->addEventToRelevantQueues(eventMsgFrag);
    
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
    CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyEventConsumer());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());
    
    QueueIDs queueIdList = eventMsgFrag.getEventConsumerTags();
    CPPUNIT_ASSERT(queueIdList.size() == 1);
    QueueIDs::const_iterator it =
      std::find(queueIdList.begin(), queueIdList.end(), qid1);
    CPPUNIT_ASSERT(it != queueIdList.end());
    CPPUNIT_ASSERT(it->index() == 0);
  }

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

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->dqmEventQueueCollection_->size() == 0);

  //
  //// DQM-specific stuff: ////
  //

  ConsumerID cid;
  edm::ParameterSet pset;

  // Consumer for HCAL:
  pset.addUntrackedParameter<std::string>("consumerName", "DQM Consumer 1");
  pset.addUntrackedParameter<std::string>("topLevelFolderName", "HCAL");
  pset.addUntrackedParameter<int>("queueSize", 10);
  pset.addUntrackedParameter<double>("consumerTimeOut", 10);
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardOld");
  boost::shared_ptr<DQMEventConsumerRegistrationInfo> ri1(
    new DQMEventConsumerRegistrationInfo(pset)
  );
  ri1->setConsumerId(++cid);
  QueueID qid1 = sharedResources_->dqmEventQueueCollection_->createQueue(ri1);
  CPPUNIT_ASSERT(qid1.isValid());
  ri1->setQueueId( qid1 );
  eventDistributor_->registerDQMEventConsumer(ri1);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(sharedResources_->dqmEventQueueCollection_->size() == 1);

  // Consumer for ECAL:
  pset.addUntrackedParameter<std::string>("consumerName", "DQM Consumer 2");
  pset.addUntrackedParameter<std::string>("topLevelFolderName", "ECAL");
  boost::shared_ptr<DQMEventConsumerRegistrationInfo> ri2(
    new DQMEventConsumerRegistrationInfo(pset)
  );
  ri2->setConsumerId(++cid);
  QueueID qid2 = sharedResources_->dqmEventQueueCollection_->createQueue(ri2);
  CPPUNIT_ASSERT(qid2.isValid());
  ri2->setQueueId( qid2 );
  eventDistributor_->registerDQMEventConsumer(ri2);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 2);
  CPPUNIT_ASSERT(sharedResources_->dqmEventQueueCollection_->size() == 2);

  // HCAL event:
  Reference* ref1 = allocate_frame_with_dqm_msg( 1111, "HCAL" );
  stor::I2OChain frag1( ref1 );
  CPPUNIT_ASSERT( frag1.messageCode() == Header::DQM_EVENT );
  eventDistributor_->addEventToRelevantQueues( frag1 );
  CPPUNIT_ASSERT( frag1.isTaggedForAnyDQMEventConsumer() );
  QueueIDs queueIdList1 = frag1.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList1.size() == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList1.begin(),queueIdList1.end(),qid1) == 1 );

  // ECAL event:
  Reference* ref2 = allocate_frame_with_dqm_msg( 2222, "ECAL" );
  stor::I2OChain frag2( ref2 );
  CPPUNIT_ASSERT( frag2.messageCode() == Header::DQM_EVENT );
  eventDistributor_->addEventToRelevantQueues( frag2 );
  CPPUNIT_ASSERT( frag2.isTaggedForAnyDQMEventConsumer() );
  QueueIDs queueIdList2 = frag2.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList2.size() == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList2.begin(),queueIdList2.end(),qid2) == 1 );

  // GT event:
  Reference* ref3 = allocate_frame_with_dqm_msg( 3333, "GT" );
  stor::I2OChain frag3( ref3 );
  CPPUNIT_ASSERT( frag3.messageCode() == Header::DQM_EVENT );
  eventDistributor_->addEventToRelevantQueues( frag3 );
  CPPUNIT_ASSERT( !frag3.isTaggedForAnyDQMEventConsumer() );
  CPPUNIT_ASSERT( frag3.getDQMEventConsumerTags().size() == 0 );

  // Wildcard consumer:
  pset.addUntrackedParameter<std::string>("consumerName", "DQM Consumer 3");
  pset.addUntrackedParameter<std::string>("topLevelFolderName", "*");
  boost::shared_ptr<DQMEventConsumerRegistrationInfo> ri3(
    new DQMEventConsumerRegistrationInfo(pset)
  );
  ri3->setConsumerId(++cid);
  QueueID qid3 = sharedResources_->dqmEventQueueCollection_->createQueue(ri3);
  CPPUNIT_ASSERT(qid3.isValid());
  ri3->setQueueId( qid3 );
  eventDistributor_->registerDQMEventConsumer(ri3);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 3);
  CPPUNIT_ASSERT(sharedResources_->dqmEventQueueCollection_->size() == 3);

  // Another HCAL event:
  Reference* ref4 = allocate_frame_with_dqm_msg( 4444, "HCAL" );
  stor::I2OChain frag4( ref4 );
  CPPUNIT_ASSERT( frag4.messageCode() == Header::DQM_EVENT );
  eventDistributor_->addEventToRelevantQueues( frag4 );
  CPPUNIT_ASSERT( frag4.isTaggedForAnyDQMEventConsumer() );
  QueueIDs queueIdList4 = frag4.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList4.size() == 2 );
  CPPUNIT_ASSERT( std::count(queueIdList4.begin(),queueIdList4.end(),qid1) == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList4.begin(),queueIdList4.end(),qid3) == 1 );

  // Another ECAL event:
  Reference* ref5 = allocate_frame_with_dqm_msg( 5555, "ECAL" );
  stor::I2OChain frag5( ref5 );
  CPPUNIT_ASSERT( frag5.messageCode() == Header::DQM_EVENT );
  eventDistributor_->addEventToRelevantQueues( frag5 );
  CPPUNIT_ASSERT( frag5.isTaggedForAnyDQMEventConsumer() );
  QueueIDs queueIdList5 = frag5.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList5.size() == 2 );
  CPPUNIT_ASSERT( std::count(queueIdList5.begin(),queueIdList5.end(),qid2) == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList5.begin(),queueIdList5.end(),qid3) == 1 );

  // Another GT event:
  Reference* ref6 = allocate_frame_with_dqm_msg( 6666, "GT" );
  stor::I2OChain frag6( ref6 );
  CPPUNIT_ASSERT( frag6.messageCode() == Header::DQM_EVENT );
  eventDistributor_->addEventToRelevantQueues( frag6 );
  CPPUNIT_ASSERT( frag6.isTaggedForAnyDQMEventConsumer() );
  QueueIDs queueIdList6 = frag6.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList6.size() == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList6.begin(),queueIdList6.end(),qid3) == 1 );
}


void testEventDistributor::testDuplicatedDQMConsumerSelection()
{
  initEventDistributor();

  CPPUNIT_ASSERT(eventDistributor_->configuredStreamCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedStreamCount() == 0);
  CPPUNIT_ASSERT(sharedResources_->initMsgCollection_->size() == 0);
  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 0);
  CPPUNIT_ASSERT(eventDistributor_->initializedConsumerCount() == 0);

  ConsumerID cid;
  edm::ParameterSet pset;

  // Consumer for HCAL:
  pset.addUntrackedParameter<std::string>("consumerName", "DQM Consumer 1");
  pset.addUntrackedParameter<std::string>("topLevelFolderName", "HCAL");
  pset.addUntrackedParameter<int>("queueSize", 10);
  pset.addUntrackedParameter<double>("consumerTimeOut", 10);
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardOld");

  boost::shared_ptr<DQMEventConsumerRegistrationInfo> ri1(
    new DQMEventConsumerRegistrationInfo(pset)
  );
  ri1->setConsumerId(++cid);
  QueueID qid1 = sharedResources_->dqmEventQueueCollection_->createQueue(ri1);
  CPPUNIT_ASSERT(qid1.isValid());
  ri1->setQueueId( qid1 );
  eventDistributor_->registerDQMEventConsumer(ri1);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 1);
  CPPUNIT_ASSERT(sharedResources_->dqmEventQueueCollection_->size() == 1);

  // re-register same consumer

  eventDistributor_->registerDQMEventConsumer(ri1);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 1);

  // 2nd consumer requesting same events
  pset.addUntrackedParameter<std::string>("consumerName", "DQM Consumer 2");
  boost::shared_ptr<DQMEventConsumerRegistrationInfo> ri2(
    new DQMEventConsumerRegistrationInfo(pset)
  );
  ri2->setConsumerId(++cid);
  QueueID qid2 = sharedResources_->dqmEventQueueCollection_->createQueue(ri2);
  CPPUNIT_ASSERT(qid2.isValid());
  ri2->setQueueId( qid2 );
  CPPUNIT_ASSERT(qid1 != qid2);
  eventDistributor_->registerDQMEventConsumer(ri2);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 2);
  CPPUNIT_ASSERT(sharedResources_->dqmEventQueueCollection_->size() == 2);

  // HCAL event:
  Reference* ref1 = allocate_frame_with_dqm_msg( 1111, "HCAL" );
  stor::I2OChain frag1( ref1 );
  CPPUNIT_ASSERT( frag1.messageCode() == Header::DQM_EVENT );
  eventDistributor_->addEventToRelevantQueues( frag1 );
  CPPUNIT_ASSERT( frag1.isTaggedForAnyDQMEventConsumer() );
  QueueIDs queueIdList1 = frag1.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList1.size() == 2 );
  CPPUNIT_ASSERT( std::count(queueIdList1.begin(),queueIdList1.end(),qid1) == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList1.begin(),queueIdList1.end(),qid2) == 1 );

  // 3rd consumer requesting same events, but different queue
  pset.addUntrackedParameter<std::string>("consumerName", "DQM Consumer 3");
  pset.addUntrackedParameter<double>("consumerTimeOut", 1024);
  boost::shared_ptr<DQMEventConsumerRegistrationInfo> ri3(
    new DQMEventConsumerRegistrationInfo(pset)
  );
  ri3->setConsumerId(++cid);
  QueueID qid3 = sharedResources_->dqmEventQueueCollection_->createQueue(ri3);
  CPPUNIT_ASSERT(qid3.isValid());
  ri3->setQueueId( qid3 );
  CPPUNIT_ASSERT(qid1 != qid3);
  CPPUNIT_ASSERT(qid2 != qid3);
  eventDistributor_->registerDQMEventConsumer(ri3);

  CPPUNIT_ASSERT(eventDistributor_->configuredConsumerCount() == 3);
  CPPUNIT_ASSERT(sharedResources_->dqmEventQueueCollection_->size() == 3);

  // Another HCAL event:
  Reference* ref2 = allocate_frame_with_dqm_msg( 4444, "HCAL" );
  stor::I2OChain frag2( ref2 );
  CPPUNIT_ASSERT( frag2.messageCode() == Header::DQM_EVENT );
  eventDistributor_->addEventToRelevantQueues( frag2 );
  CPPUNIT_ASSERT( frag2.isTaggedForAnyDQMEventConsumer() );
  QueueIDs queueIdList2 = frag2.getDQMEventConsumerTags();
  CPPUNIT_ASSERT( queueIdList2.size() == 3 );
  CPPUNIT_ASSERT( std::count(queueIdList2.begin(),queueIdList2.end(),qid1) == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList2.begin(),queueIdList2.end(),qid2) == 1 );
  CPPUNIT_ASSERT( std::count(queueIdList2.begin(),queueIdList2.end(),qid3) == 1 );
}



// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testEventDistributor);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
