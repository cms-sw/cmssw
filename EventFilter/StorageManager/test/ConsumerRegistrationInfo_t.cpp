#include <iostream>
#include <iomanip>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"

using namespace stor;

class testConsumerRegistrationInfo : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testConsumerRegistrationInfo);
  CPPUNIT_TEST(testEventConsumerRegistrationInfo);
  CPPUNIT_TEST(testIdenticalEventConsumers);
  CPPUNIT_TEST(testDQMEventConsumerRegistrationInfo);
  CPPUNIT_TEST(testIdenticalDQMEventConsumers);

  CPPUNIT_TEST_SUITE_END();

public:

  void testEventConsumerRegistrationInfo();
  void testIdenticalEventConsumers();

  void testDQMEventConsumerRegistrationInfo();
  void testIdenticalDQMEventConsumers();

};


void testConsumerRegistrationInfo::testEventConsumerRegistrationInfo()
{
  Strings eventSelection;
  eventSelection.push_back( "DQM1" );
  eventSelection.push_back( "DQM2" );

  const std::string triggerSelection = "DQM1 || DQM2";

  QueueID qid(enquing_policy::DiscardOld, 3);

  EventConsumerRegistrationInfo ecri(
    "Test Consumer",
    triggerSelection,
    eventSelection,
    "hltOutputDQM",
    1,
    false,
    qid.index(),
    qid.policy(),
    boost::posix_time::seconds(10),
    "localhost"
  );
  ecri.setQueueId( qid );

  CPPUNIT_ASSERT( ecri.consumerName() == "Test Consumer" );
  CPPUNIT_ASSERT( ecri.consumerId() == ConsumerID(0) );
  CPPUNIT_ASSERT( ecri.queueId() == qid );
  CPPUNIT_ASSERT( ecri.queueSize() == 3 );
  CPPUNIT_ASSERT( ecri.queuePolicy() == enquing_policy::DiscardOld );
  CPPUNIT_ASSERT( ecri.secondsToStale() == boost::posix_time::seconds(10) );
  CPPUNIT_ASSERT( ecri.triggerSelection() == triggerSelection );
  CPPUNIT_ASSERT( ecri.eventSelection() == eventSelection );
  CPPUNIT_ASSERT( ecri.outputModuleLabel() == "hltOutputDQM" );
  CPPUNIT_ASSERT( ecri.uniqueEvents() == false );
  CPPUNIT_ASSERT( ecri.isProxyServer() == false );
  CPPUNIT_ASSERT( ecri.remoteHost() == "localhost" );
}


void testConsumerRegistrationInfo::testIdenticalEventConsumers()
{
  Strings eventSelection;
  eventSelection.push_back( "DQM1" );
  eventSelection.push_back( "DQM2" );

  const std::string triggerSelection = "DQM1 || DQM2";

  QueueID qid(enquing_policy::DiscardOld, 3);

  EventConsumerRegistrationInfo ecri1(
    "Consumer A",
    triggerSelection,
    eventSelection,
    "hltOutputDQM",
    1,
    false,
    qid.index(),
    qid.policy(),
    boost::posix_time::seconds(10),
    "localhost"
  );
  ecri1.setQueueId( qid );

  EventConsumerRegistrationInfo ecri2(
    "Consumer B",
    triggerSelection,
    eventSelection,
    "hltOutputDQM",
    1,
    false,
    qid.index(),
    qid.policy(),
    boost::posix_time::seconds(10),
    "remotehost"
  );
  ecri2.setQueueId( qid );

  const std::string triggerSelection2 = "DQM1";

  EventConsumerRegistrationInfo ecri3(
    "Consumer C",
    triggerSelection2,
    eventSelection,
    "hltOutputDQM",
    1,
    false,
    qid.index(),
    qid.policy(),
    boost::posix_time::seconds(10),
    "farawayhost"
  );
  ecri3.setQueueId( qid );

  EventConsumerRegistrationInfo ecri4(
    "Consumer D",
    triggerSelection2,
    eventSelection,
    "hltOutputDQM",
    1,
    true, // unique events
    qid.index(),
    qid.policy(),
    boost::posix_time::seconds(10),
    "inanothergalaxyhost"
  );
  ecri4.setQueueId( qid );

  EventConsumerRegistrationInfo ecri5(
    "Consumer D",
    triggerSelection2,
    eventSelection,
    "hltOutputDQM",
    10, //different prescale
    true, // unique events
    qid.index(),
    qid.policy(),
    boost::posix_time::seconds(10),
    "inanotheruniversehost"
  );
  ecri5.setQueueId( qid );

  CPPUNIT_ASSERT( ecri1 == ecri2 );
  CPPUNIT_ASSERT( ecri1 != ecri3 );
  CPPUNIT_ASSERT( ecri2 != ecri3 );
  CPPUNIT_ASSERT( ecri3 != ecri4 );
  CPPUNIT_ASSERT( ecri4 != ecri5 );
}


void testConsumerRegistrationInfo::testDQMEventConsumerRegistrationInfo()
{
  QueueID qid(stor::enquing_policy::DiscardNew, 2);
  DQMEventConsumerRegistrationInfo ecri(
    "Histo Consumer",
    "*",
    qid.index(),
    qid.policy(),
    boost::posix_time::seconds(1024),
    "localhost"
  );
  ecri.setQueueId( qid );
  
  CPPUNIT_ASSERT( ecri.consumerName() == "Histo Consumer" );
  CPPUNIT_ASSERT( ecri.consumerId() == ConsumerID(0) );
  CPPUNIT_ASSERT( ecri.queueId() == qid );
  CPPUNIT_ASSERT( ecri.queueSize() == 2 );
  CPPUNIT_ASSERT( ecri.queuePolicy() == enquing_policy::DiscardNew );
  CPPUNIT_ASSERT( ecri.secondsToStale() == boost::posix_time::seconds(1024) );
  CPPUNIT_ASSERT( ecri.topLevelFolderName() == "*" );
  CPPUNIT_ASSERT( ecri.isProxyServer() == false );
  CPPUNIT_ASSERT( ecri.remoteHost() == "localhost" );
}


void testConsumerRegistrationInfo::testIdenticalDQMEventConsumers()
{
  QueueID qid(stor::enquing_policy::DiscardNew, 2);
  DQMEventConsumerRegistrationInfo ecri1(
    "Histo Consumer 1",
    "*",
    qid.index(),
    qid.policy(),
    boost::posix_time::seconds(1024),
    "localhost"
  );
  ecri1.setQueueId( qid );

  DQMEventConsumerRegistrationInfo ecri2(
    "Histo Consumer 2",
    "*",
    qid.index(),
    qid.policy(),
    boost::posix_time::seconds(1024),
    "remotehost"
  );
  ecri2.setQueueId( qid );

  DQMEventConsumerRegistrationInfo ecri3(
    "Histo Consumer 3",
    "HCAL",
    qid.index(),
    qid.policy(),
    boost::posix_time::seconds(1024),
    "farawayhost"
  );
  ecri3.setQueueId( qid );

  DQMEventConsumerRegistrationInfo ecri4(
    "Histo Consumer 3",
    "HCAL",
    qid.index(),
    qid.policy(),
    boost::posix_time::seconds(10),
    "farawayhost"
  );
  ecri4.setQueueId( qid );
 
  CPPUNIT_ASSERT( ecri1 == ecri2 );
  CPPUNIT_ASSERT( ecri1 != ecri3 );
  CPPUNIT_ASSERT( ecri2 != ecri3 );
  CPPUNIT_ASSERT( ecri3 != ecri4 );
}


// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testConsumerRegistrationInfo);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
