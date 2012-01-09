#include <iostream>
#include <iomanip>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"

using namespace stor;

class testConsumerRegistrationInfo : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testConsumerRegistrationInfo);
  CPPUNIT_TEST(testEventConsumerRegistrationInfo);
  CPPUNIT_TEST(testEventConsumerPSet);
  CPPUNIT_TEST(testIncompleteEventConsumerPSet);
  CPPUNIT_TEST(testIdenticalEventConsumers);
  CPPUNIT_TEST(testDQMEventConsumerRegistrationInfo);
  CPPUNIT_TEST(testIdenticalDQMEventConsumers);

  CPPUNIT_TEST_SUITE_END();

public:

  void testEventConsumerRegistrationInfo();
  void testEventConsumerPSet();
  void testIncompleteEventConsumerPSet();
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

  edm::ParameterSet pset;
  pset.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  pset.addUntrackedParameter<std::string>("TriggerSelector", triggerSelection);
  pset.addParameter<Strings>("TrackedEventSelection", eventSelection);
  pset.addUntrackedParameter<bool>("uniqueEvents", false);
  pset.addUntrackedParameter<int>("prescale", 5);
  pset.addUntrackedParameter<int>("headerRetryInterval", 11);
  pset.addUntrackedParameter<int>("maxConnectTries", 13);
  pset.addUntrackedParameter<int>("connectTrySleepTime", 4);
  pset.addUntrackedParameter<double>("maxEventRequestRate", 10);
  pset.addUntrackedParameter<std::string>("consumerName", "Test Consumer");
  pset.addParameter<std::string>("sourceURL", "mySource");
  pset.addUntrackedParameter<int>("queueSize", 9);
  pset.addUntrackedParameter<double>("consumerTimeOut", 14);
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardNew");

  EventConsumerRegistrationInfo ecri(pset, "localhost");
  ecri.setQueueId( qid );

  CPPUNIT_ASSERT( ecri.consumerName() == "Test Consumer" );
  CPPUNIT_ASSERT( ecri.consumerId() == ConsumerID(0) );
  CPPUNIT_ASSERT( ecri.queueId() == qid );
  CPPUNIT_ASSERT( ecri.queueSize() == 9 );
  CPPUNIT_ASSERT( ecri.queuePolicy() == enquing_policy::DiscardNew );
  CPPUNIT_ASSERT( ecri.secondsToStale() == boost::posix_time::seconds(14) );
  CPPUNIT_ASSERT( ecri.triggerSelection() == triggerSelection );
  CPPUNIT_ASSERT( ecri.eventSelection() == eventSelection );
  CPPUNIT_ASSERT( ecri.outputModuleLabel() == "hltOutputDQM" );
  CPPUNIT_ASSERT( ecri.uniqueEvents() == false );
  CPPUNIT_ASSERT( ecri.remoteHost() == "localhost" );
  CPPUNIT_ASSERT( ecri.sourceURL() == "mySource" );
  CPPUNIT_ASSERT( ecri.headerRetryInterval() == 11 );
  CPPUNIT_ASSERT( ecri.maxConnectTries() == 13 );
  CPPUNIT_ASSERT( ecri.connectTrySleepTime() == 4 );
  CPPUNIT_ASSERT( ecri.minEventRequestInterval() == boost::posix_time::milliseconds(100) );
  CPPUNIT_ASSERT( fabs(stor::utils::durationToSeconds(ecri.minEventRequestInterval()) - 0.1) < 0.0001);

  edm::ParameterSet ecriPSet = ecri.getPSet();
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<std::string>("SelectHLTOutput") == "hltOutputDQM" );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<std::string>("TriggerSelector") == triggerSelection );
  CPPUNIT_ASSERT( ecriPSet.getParameter<Strings>("TrackedEventSelection") == eventSelection );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<bool>("uniqueEvents") == false );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<int>("prescale") == 5 );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<int>("headerRetryInterval") == 11 );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<int>("maxConnectTries") == 13 );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<int>("connectTrySleepTime") == 4 );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<double>("maxEventRequestRate") == 10 );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<std::string>("consumerName") == "Test Consumer" );
  CPPUNIT_ASSERT( ecriPSet.getParameter<std::string>("sourceURL") == "mySource" );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<int>("queueSize") == 9 );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<double>("consumerTimeOut") == 14 );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<std::string>("queuePolicy") == "DiscardNew" );
}


void testConsumerRegistrationInfo::testEventConsumerPSet()
{
  Strings eventSelection;
  eventSelection.push_back( "DQM" );
  eventSelection.push_back( "HLT" );

  const std::string triggerSelection = "HLT || DQM";

  edm::ParameterSet origPSet;
  origPSet.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  origPSet.addUntrackedParameter<std::string>("TriggerSelector", triggerSelection);
  origPSet.addParameter<Strings>("TrackedEventSelection", eventSelection);
  origPSet.addUntrackedParameter<bool>("uniqueEvents", true);
  origPSet.addUntrackedParameter<int>("prescale", 5);
  origPSet.addUntrackedParameter<int>("queueSize", 10);
  origPSet.addUntrackedParameter<double>("consumerTimeOut", 33);
  origPSet.addUntrackedParameter<std::string>("queuePolicy", "DiscardNew");
  origPSet.addUntrackedParameter<double>("maxEventRequestRate", 25);
  origPSet.addUntrackedParameter<int>("headerRetryInterval", 11);
  origPSet.addUntrackedParameter<int>("maxConnectTries", 13);
  origPSet.addUntrackedParameter<int>("connectTrySleepTime", 4);
  origPSet.addUntrackedParameter<std::string>("consumerName", "Test Consumer");
  origPSet.addParameter<std::string>("sourceURL", "mySource");

  EventConsumerRegistrationInfo ecri(origPSet, "localhost");

  edm::ParameterSet ecriPSet = ecri.getPSet();
  CPPUNIT_ASSERT( isTransientEqual(origPSet, ecriPSet) );
}


void testConsumerRegistrationInfo::testIncompleteEventConsumerPSet()
{
  edm::ParameterSet origPSet;
  origPSet.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  
  EventConsumerRegistrationInfo ecri(origPSet, "localhost");

  CPPUNIT_ASSERT( ecri.queueSize() == 0 );
  CPPUNIT_ASSERT( ecri.queuePolicy() == enquing_policy::Max );
  CPPUNIT_ASSERT( ecri.secondsToStale() == boost::posix_time::seconds(0) );
  CPPUNIT_ASSERT( ecri.minEventRequestInterval() == boost::posix_time::not_a_date_time );
  CPPUNIT_ASSERT( ecri.consumerName() == "Unknown" );
  CPPUNIT_ASSERT( ecri.sourceURL() == "Unknown" );
  CPPUNIT_ASSERT( ecri.headerRetryInterval() == 5);
  CPPUNIT_ASSERT( ecri.maxConnectTries() == 300);
  CPPUNIT_ASSERT( ecri.connectTrySleepTime() == 10);

  edm::ParameterSet ecriPSet = ecri.getPSet();
  CPPUNIT_ASSERT( ! isTransientEqual(origPSet, ecriPSet) );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<std::string>("SelectHLTOutput") == "hltOutputDQM" );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<std::string>("TriggerSelector") == "" );
  CPPUNIT_ASSERT( ecriPSet.getParameter<Strings>("TrackedEventSelection") == Strings() );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<bool>("uniqueEvents") == false );
  CPPUNIT_ASSERT( ecriPSet.getUntrackedParameter<int>("prescale") == 1 );
  CPPUNIT_ASSERT( ! ecriPSet.exists("queueSize") );
  CPPUNIT_ASSERT( ! ecriPSet.exists("consumerTimeOut") );
  CPPUNIT_ASSERT( ! ecriPSet.exists("queuePolicy") );
  CPPUNIT_ASSERT( ! ecriPSet.exists("maxEventRequestRate") );
  CPPUNIT_ASSERT( ! ecriPSet.exists("consumerName") );
  CPPUNIT_ASSERT( ! ecriPSet.exists("sourceURL") );
  CPPUNIT_ASSERT( ! ecriPSet.exists("headerRetryInterval") );
  CPPUNIT_ASSERT( ! ecriPSet.exists("maxConnectTries") );
  CPPUNIT_ASSERT( ! ecriPSet.exists("connectTrySleepTime") );

  EventServingParams defaults;
  defaults.activeConsumerTimeout_ =  boost::posix_time::seconds(12);
  defaults.consumerQueueSize_ = 22;
  defaults.consumerQueuePolicy_ = "DiscardOld";

  EventConsumerRegistrationInfo ecriDefaults(origPSet, defaults);

  CPPUNIT_ASSERT( ecriDefaults.queueSize() == defaults.consumerQueueSize_ );
  CPPUNIT_ASSERT( ecriDefaults.queuePolicy() == enquing_policy::DiscardOld );
  CPPUNIT_ASSERT( ecriDefaults.secondsToStale() == defaults.activeConsumerTimeout_ );
  CPPUNIT_ASSERT( ecriDefaults.minEventRequestInterval() == boost::posix_time::not_a_date_time );

  edm::ParameterSet ecriDefaultsPSet = ecriDefaults.getPSet();
  CPPUNIT_ASSERT( ! isTransientEqual(origPSet, ecriDefaultsPSet) );
  CPPUNIT_ASSERT( ! isTransientEqual(ecriPSet, ecriDefaultsPSet) );
  CPPUNIT_ASSERT( ecriDefaultsPSet.getUntrackedParameter<std::string>("SelectHLTOutput") == "hltOutputDQM" );
  CPPUNIT_ASSERT( ecriDefaultsPSet.getUntrackedParameter<std::string>("TriggerSelector") == "" );
  CPPUNIT_ASSERT( ecriDefaultsPSet.getParameter<Strings>("TrackedEventSelection") == Strings() );
  CPPUNIT_ASSERT( ecriDefaultsPSet.getUntrackedParameter<bool>("uniqueEvents") == false );
  CPPUNIT_ASSERT( ecriDefaultsPSet.getUntrackedParameter<int>("prescale") == 1 );
  CPPUNIT_ASSERT( ecriDefaultsPSet.getUntrackedParameter<int>("queueSize") == 22 );
  CPPUNIT_ASSERT( ecriDefaultsPSet.getUntrackedParameter<double>("consumerTimeOut") == 12 );
  CPPUNIT_ASSERT( ecriDefaultsPSet.getUntrackedParameter<std::string>("queuePolicy") == "DiscardOld" );
  CPPUNIT_ASSERT( ! ecriDefaultsPSet.exists("maxEventRequestRate") );
  CPPUNIT_ASSERT( ! ecriDefaultsPSet.exists("consumerName") );
  CPPUNIT_ASSERT( ! ecriDefaultsPSet.exists("sourceURL") );
  CPPUNIT_ASSERT( ! ecriDefaultsPSet.exists("headerRetryInterval") );
  CPPUNIT_ASSERT( ! ecriDefaultsPSet.exists("maxConnectTries") );
  CPPUNIT_ASSERT( ! ecriDefaultsPSet.exists("connectTrySleepTime") );
}


void testConsumerRegistrationInfo::testIdenticalEventConsumers()
{
  const std::string triggerSelection = "DQM1 || DQM2";

  QueueID qid(enquing_policy::DiscardOld, 3);
 
  edm::ParameterSet pset1;
  pset1.addUntrackedParameter<std::string>("consumerName", "Consumer A");
  pset1.addParameter<std::string>("sourceURL", "mySource");
  pset1.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  pset1.addUntrackedParameter<std::string>("TriggerSelector", triggerSelection);
  pset1.addUntrackedParameter<bool>("uniqueEvents", false);

  EventConsumerRegistrationInfo ecri1(pset1, "localhost");
  ecri1.setQueueId( qid );

  edm::ParameterSet pset2 = pset1;
  pset2.addUntrackedParameter<std::string>("consumerName", "Consumer B");

  EventConsumerRegistrationInfo ecri2(pset2, "remotehost");
  ecri2.setQueueId( qid );

  const std::string triggerSelection2 = "DQM1";
  edm::ParameterSet pset3 = pset1;
  pset3.addUntrackedParameter<std::string>("consumerName", "Consumer C");
  pset3.addUntrackedParameter<std::string>("TriggerSelector", triggerSelection2);
  EventConsumerRegistrationInfo ecri3(pset3, "farawayhost");
  ecri3.setQueueId( qid );

  edm::ParameterSet pset4 = pset3;
  pset4.addUntrackedParameter<std::string>("consumerName", "Consumer D");
  pset4.addUntrackedParameter<bool>("uniqueEvents", true);
  EventConsumerRegistrationInfo ecri4(pset4, "inanothergalaxyhost");
  ecri4.setQueueId( qid );

  edm::ParameterSet pset5 = pset4;
  pset5.addUntrackedParameter<std::string>("consumerName", "Consumer E");
  pset5.addUntrackedParameter<int>("prescale", 10);
  EventConsumerRegistrationInfo ecri5(pset5, "inanotheruniversehost");
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

  edm::ParameterSet pset;
  pset.addUntrackedParameter<std::string>("consumerName", "Histo Consumer");
  pset.addParameter<std::string>("sourceURL", "myDQMSource");
  pset.addUntrackedParameter<int>("queueSize", 6);
  pset.addUntrackedParameter<double>("consumerTimeOut", 1024);
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardNew");
  pset.addUntrackedParameter<std::string>("topLevelFolderName", "*");
  pset.addUntrackedParameter<int>("maxConnectTries", 11);
  pset.addUntrackedParameter<int>("connectTrySleepTime", 6);
  pset.addUntrackedParameter<int>("retryInterval", 21);

  DQMEventConsumerRegistrationInfo dcri(pset, "localhost");
  dcri.setQueueId( qid );
  
  CPPUNIT_ASSERT( dcri.consumerName() == "Histo Consumer" );
  CPPUNIT_ASSERT( dcri.consumerId() == ConsumerID(0) );
  CPPUNIT_ASSERT( dcri.queueId() == qid );
  CPPUNIT_ASSERT( dcri.queueSize() == 6 );
  CPPUNIT_ASSERT( dcri.queuePolicy() == enquing_policy::DiscardNew );
  CPPUNIT_ASSERT( dcri.secondsToStale() == boost::posix_time::seconds(1024) );
  CPPUNIT_ASSERT( dcri.topLevelFolderName() == "*" );
  CPPUNIT_ASSERT( dcri.remoteHost() == "localhost" );
  CPPUNIT_ASSERT( dcri.sourceURL() == "myDQMSource" );
  CPPUNIT_ASSERT( dcri.retryInterval() == 21 );
  CPPUNIT_ASSERT( dcri.maxConnectTries() == 11 );
  CPPUNIT_ASSERT( dcri.connectTrySleepTime() == 6 );
}


void testConsumerRegistrationInfo::testIdenticalDQMEventConsumers()
{
  QueueID qid(stor::enquing_policy::DiscardNew, 2);

  edm::ParameterSet pset;
  pset.addUntrackedParameter<std::string>("consumerName", "Histo Consumer 1");
  pset.addUntrackedParameter<std::string>("topLevelFolderName", "*");
  DQMEventConsumerRegistrationInfo dcri1(pset, "localhost");
  dcri1.setQueueId( qid );

  pset.addUntrackedParameter<std::string>("consumerName", "Histo Consumer 2");
  DQMEventConsumerRegistrationInfo dcri2(pset, "remotehost");
  dcri2.setQueueId( qid );

  pset.addUntrackedParameter<std::string>("consumerName", "Histo Consumer 3");
  pset.addUntrackedParameter<std::string>("topLevelFolderName", "HCAL");
  DQMEventConsumerRegistrationInfo dcri3(pset, "farawayhost");
  dcri3.setQueueId( qid );

  pset.addUntrackedParameter<std::string>("consumerName", "Histo Consumer 4");
  pset.addUntrackedParameter<double>("consumerTimeOut", 10);
  DQMEventConsumerRegistrationInfo dcri4(pset, "farawayhost");
  dcri4.setQueueId( qid );
 
  CPPUNIT_ASSERT( dcri1 == dcri2 );
  CPPUNIT_ASSERT( dcri1 != dcri3 );
  CPPUNIT_ASSERT( dcri2 != dcri3 );
  CPPUNIT_ASSERT( dcri3 != dcri4 );
}


// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testConsumerRegistrationInfo);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
