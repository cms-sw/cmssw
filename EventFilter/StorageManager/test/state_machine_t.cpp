#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <iostream>
#include <map>
#include <list>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include <boost/statechart/event_base.hpp>
#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/CommandQueue.h"
#include "EventFilter/StorageManager/interface/DiscardManager.h"
#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/FragmentQueue.h"
#include "EventFilter/StorageManager/interface/FragmentStore.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/RegistrationCollection.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/TransitionRecord.h"
#include "EventFilter/StorageManager/test/MockApplication.h"
#include "EventFilter/StorageManager/test/MockDiskWriterResources.h"
#include "EventFilter/StorageManager/test/MockDQMEventProcessorResources.h"
#include "EventFilter/StorageManager/test/MockNotifier.h"

using namespace std;
using namespace boost::statechart;
using namespace stor;

/////////////////////////////////////////////////////////////
//
// This test exercises the state machine
//
/////////////////////////////////////////////////////////////

class testStateMachine : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testStateMachine);
  CPPUNIT_TEST(testConstructed);
  CPPUNIT_TEST(testHalted);
  CPPUNIT_TEST(testStopped);
  CPPUNIT_TEST(testProcessing);
  CPPUNIT_TEST(testFail);
  CPPUNIT_TEST(testEnableSequence);
  CPPUNIT_TEST(testStopSequence);
  CPPUNIT_TEST(testHaltSequence);
  CPPUNIT_TEST(testReconfigureSequence);
  CPPUNIT_TEST(testEmergencyStopSequence);
  CPPUNIT_TEST(testAllStatesGoToFailed);

  CPPUNIT_TEST_SUITE_END();

private:
  // Typedefs:
  typedef std::map<std::string, boost::shared_ptr<event_base> > EventMap;
  typedef std::vector<std::string> EventList;
  typedef std::vector<TransitionRecord> TransitionList;

public:
  void setUp();
  void tearDown();

  void testConstructed();
  void testHalted();
  void testStopped();
  void testProcessing();
  void testFail();
  void testEnableSequence();
  void testStopSequence();
  void testHaltSequence();
  void testReconfigureSequence();
  void testEmergencyStopSequence();
  void testAllStatesGoToFailed();

private:
  void resetStateMachine();
  void processEvent( stor::EventPtr_t requestedEvent );
  bool checkState( const std::string& expected );
  void checkSignals( const EventList& elist, const std::string& expected);
  bool checkHistory( const TransitionList& steps );

  static xdaq::Application* app_;
  EventDistributor* ed_;
  FragmentStore* fs_;
  MockNotifier* mn_;

  StateMachine *machine_;
  static SharedResourcesPtr sr_;
};

xdaq::Application* testStateMachine::app_;
SharedResourcesPtr testStateMachine::sr_;

void testStateMachine::setUp()
{
  // 30-Jun-2009, KAB - to avoid the problem in which we try to re-declare infospace
  // variables to the MockApplication infospace for each test, we need to create just
  // one mock application and one shared resources instance for all tests.
  if ( sr_.get() == 0 )
  {
    mkdir("/tmp/log", 644); // dummy dir to avoid DiskSpaceAlarms
    app_ = mockapps::getMockXdaqApplication();

    sr_.reset(new SharedResources());
    sr_->configuration_.reset(new Configuration(app_->getApplicationInfoSpace(), 0));
    sr_->initMsgCollection_.reset(new InitMsgCollection());
    sr_->diskWriterResources_.reset(new MockDiskWriterResources());
    sr_->dqmEventProcessorResources_.reset(new MockDQMEventProcessorResources());
    sr_->commandQueue_.reset(new CommandQueue(32));
    sr_->fragmentQueue_.reset(new FragmentQueue(32));
    sr_->registrationQueue_.reset(new RegistrationQueue(32));
    sr_->streamQueue_.reset(new StreamQueue(32));
    sr_->dqmEventQueue_.reset(new DQMEventQueue(32));
    sr_->statisticsReporter_.reset( new StatisticsReporter( app_, sr_ ) );
    EventConsumerMonitorCollection& ecmc = 
      sr_->statisticsReporter_->getEventConsumerMonitorCollection();
    sr_->eventQueueCollection_.reset( new EventQueueCollection( ecmc ) );
    DQMConsumerMonitorCollection& dcmc = 
      sr_->statisticsReporter_->getDQMConsumerMonitorCollection();
    sr_->dqmEventQueueCollection_.reset( new DQMEventQueueCollection( dcmc ) );
    
    sr_->discardManager_.reset(new DiscardManager(app_->getApplicationContext(),
                                                  app_->getApplicationDescriptor(),
                                                  sr_->statisticsReporter_->getDataSenderMonitorCollection()));

    sr_->registrationCollection_.reset(new RegistrationCollection());
  }

  ed_ = new EventDistributor(sr_);
  fs_ = new FragmentStore(1);
  mn_ = new MockNotifier( app_ );

  machine_ = new StateMachine( ed_, fs_, mn_, sr_ );
}

void testStateMachine::tearDown()
{
  //delete machine_;
}

void testStateMachine::resetStateMachine()
{
  sr_->statisticsReporter_->
    getStateMachineMonitorCollection().reset(boost::posix_time::not_a_date_time);
  machine_->initiate();
}

/////////////////////////////////////////////////////////////////////
//// Simulate the processing of events by the fragment processor ////
/////////////////////////////////////////////////////////////////////
void testStateMachine::processEvent( stor::EventPtr_t requestedEvent )
{
  boost::shared_ptr<CommandQueue> cmdQueue;
  cmdQueue = machine_->getSharedResources()->commandQueue_;

  cmdQueue->enqWait( requestedEvent );
  
  stor::EventPtr_t nextEvent;
  while ( cmdQueue->deqNowait( nextEvent ) )
  {
    machine_->process_event( *nextEvent );
    machine_->getCurrentState().noFragmentToProcess();
  }
}


////////////////////////////////////////////
//// Returns false if unexpected state: ////
////////////////////////////////////////////
bool testStateMachine::checkState( const std::string& expected )
{
  const std::string actual = machine_->getCurrentStateName();
  if( actual != expected )
  {
    std::cerr << "Expecting " << expected << ", got " << actual << std::endl;
    sr_->statisticsReporter_->
      getStateMachineMonitorCollection().dumpHistory( std::cerr );
    return false;
  }
  return true;
}

/////////////////////////////////////////////////////////////
//// Check if every signal not on the list gets ignored: ////
/////////////////////////////////////////////////////////////
void testStateMachine::checkSignals
(
  const EventList& elist,
  const std::string& expected
)
{
  EventMap emap;
  emap[ "Configure" ] = boost::shared_ptr<event_base>( new Configure() );
  emap[ "Enable" ] = boost::shared_ptr<event_base>( new Enable() );
  emap[ "Stop" ] = boost::shared_ptr<event_base>( new Stop() );
  emap[ "Halt" ] = boost::shared_ptr<event_base>( new Halt() );
  emap[ "Reconfigure" ] = boost::shared_ptr<event_base>( new Reconfigure() );
  emap[ "EmergencyStop" ] = boost::shared_ptr<event_base>( new EmergencyStop() );
  emap[ "QueuesEmpty" ] = boost::shared_ptr<event_base>( new QueuesEmpty() );
  emap[ "StopDone" ] = boost::shared_ptr<event_base>( new StopDone() );
  emap[ "HaltDone" ] = boost::shared_ptr<event_base>( new HaltDone() );
  emap[ "StartRun" ] = boost::shared_ptr<event_base>( new StartRun() );
  emap[ "EndRun" ] = boost::shared_ptr<event_base>( new EndRun() );
  emap[ "Fail" ] = boost::shared_ptr<event_base>( new Fail() );

  for ( EventMap::const_iterator it = emap.begin(), itEnd = emap.end();
        it != itEnd; ++it )
  {
    if ( std::find( elist.begin(), elist.end(), it->first ) == elist.end() )
    {
      processEvent( it->second );
      CPPUNIT_ASSERT( checkState( expected ) );
    }
  }
}


////////////////////////////////////////////////////////////////////
//// Check if history matches expected sequence of transitions: ////
////////////////////////////////////////////////////////////////////
bool testStateMachine::checkHistory( const TransitionList& steps )
{

  StateMachineMonitorCollection::History h; 
  sr_->statisticsReporter_->
    getStateMachineMonitorCollection().getHistory( h );
  const unsigned int hsize = h.size();
  const unsigned int ssize = steps.size();
  
  bool ok = true;
  
  if( ssize != hsize )
  {
    ok = false;
  }
  
  if( ok )
  {
    for( unsigned int i = 0; i < hsize; ++i )
    {
      if( h[i].isEntry() != steps[i].isEntry() ||
        h[i].stateName() != steps[i].stateName() )
      {
        ok = false;
        break;
      }
    }
  }
  
  if( !ok )
  {
    std::cerr << "**** History mismatch ****" << std::endl;
    std::cerr << "Actual:" << std::endl;
    for( unsigned int i = 0; i < hsize; ++i )
    {
      if( h[i].isEntry() )
      {
        std::cerr << " entered ";
      }
      else
      {
        std::cerr << " exited ";
      }
      std::cerr << h[i].stateName() << std::endl;
    }
    std::cerr << "Expected:" << std::endl;
    for( unsigned int j = 0; j < ssize; ++j )
    {
      if( steps[j].isEntry() )
      {
        std::cerr << " entered ";
      }
      else
      {
        std::cerr << " exited ";
      }
      std::cerr << steps[j].stateName() << std::endl;
    }
  }
  return ok;
}


void testStateMachine::testConstructed()
{
  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  std::cout << std::endl << "**** Testing illegal signals in Constructed state ****" << std::endl;

  EventList elist;
  elist.push_back( "Configure" );
  elist.push_back( "Fail" );

  checkSignals( elist, "Constructed" );

  machine_->terminate();
}


void testStateMachine::testHalted()
{
  stor::EventPtr_t stMachEvent;

  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  stMachEvent.reset( new Halt() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Halted" ) );

  std::cout << std::endl << "**** Testing illegal signals in Halted state ****" << std::endl;

  EventList elist;
  elist.push_back( "Configure" );
  elist.push_back( "Fail" );

  checkSignals( elist, "Halted" );

  machine_->terminate();
}

void testStateMachine::testStopped()
{
  stor::EventPtr_t stMachEvent;

  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  std::cout << std::endl << "**** Testing illegal signals in Stopped state ****" << std::endl;

  EventList elist;
  elist.push_back( "Reconfigure" );
  elist.push_back( "Enable" );
  elist.push_back( "Halt" );
  elist.push_back( "Fail" );

  checkSignals( elist, "Stopped" );

  stMachEvent.reset( new Halt() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Halted" ) );

  machine_->terminate();
}

void testStateMachine::testProcessing()
{
  stor::EventPtr_t stMachEvent;

  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  stMachEvent.reset( new Enable() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Processing" ) );

  std::cout << std::endl << "**** Testing illegal signals in Processing state ****" << std::endl;

  EventList elist;
  elist.push_back( "Halt" );
  elist.push_back( "Stop" );
  elist.push_back( "EmergencyStop" );
  elist.push_back( "Fail" );

  checkSignals( elist, "Processing" );

  stMachEvent.reset( new Halt() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Halted" ) );

  machine_->terminate();
}


void testStateMachine::testFail()
{
  stor::EventPtr_t stMachEvent;

  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Fail() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Failed" ) );

  std::cout << std::endl << "**** Making sure no signal changes Failed state ****" << std::endl;

  EventList elist;
  checkSignals( elist, "Failed" );

  machine_->terminate();
}


void testStateMachine::testEnableSequence()
{
  stor::EventPtr_t stMachEvent;

  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  stMachEvent.reset( new Enable() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Processing" ) );

  std::cout << std::endl << "**** Testing if Enable does the right sequence ****" << std::endl;

  TransitionList steps;
  steps.push_back( TransitionRecord( "Normal", true ) );
  steps.push_back( TransitionRecord( "Constructed", true ) );
  steps.push_back( TransitionRecord( "Constructed", false ) );
  steps.push_back( TransitionRecord( "Ready", true ) );
  steps.push_back( TransitionRecord( "Stopped", true ) );
  steps.push_back( TransitionRecord( "Stopped", false ) );
  steps.push_back( TransitionRecord( "Enabled", true ) );
  steps.push_back( TransitionRecord( "Starting", true ) );
  steps.push_back( TransitionRecord( "Starting", false ) );
  steps.push_back( TransitionRecord( "Running", true ) );
  steps.push_back( TransitionRecord( "Processing", true ) );
  CPPUNIT_ASSERT( checkHistory( steps ) );

  machine_->terminate();
}

void testStateMachine::testStopSequence()
{
  stor::EventPtr_t stMachEvent;

  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  stMachEvent.reset( new Enable() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Processing" ) );

  sr_->statisticsReporter_->
    getStateMachineMonitorCollection().reset(boost::posix_time::not_a_date_time);
  stMachEvent.reset( new Stop() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  std::cout << std::endl << "**** Testing if Stopping goes through the right sequence ****" << std::endl;

  TransitionList steps;
  steps.push_back( TransitionRecord( "Processing", false ) );
  steps.push_back( TransitionRecord( "DrainingQueues", true ) );
  steps.push_back( TransitionRecord( "DrainingQueues", false ) );
  steps.push_back( TransitionRecord( "FinishingDQM", true ) );
  steps.push_back( TransitionRecord( "FinishingDQM", false ) );
  steps.push_back( TransitionRecord( "Running", false ) );
  steps.push_back( TransitionRecord( "Stopping", true ) );
  steps.push_back( TransitionRecord( "Stopping", false ) );
  steps.push_back( TransitionRecord( "Enabled", false ) );
  steps.push_back( TransitionRecord( "Stopped", true ) );
  CPPUNIT_ASSERT( checkHistory( steps ) );

  machine_->terminate();
}

void testStateMachine::testHaltSequence()
{
  stor::EventPtr_t stMachEvent;

  resetStateMachine();
  sr_->statisticsReporter_->
    getStateMachineMonitorCollection().reset(boost::posix_time::not_a_date_time);
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  stMachEvent.reset( new Enable() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Processing" ) );

  sr_->statisticsReporter_->
    getStateMachineMonitorCollection().reset(boost::posix_time::not_a_date_time);
  stMachEvent.reset( new Halt() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Halted" ) );

  std::cout << std::endl << "**** Testing if Halt does the right sequence ****" << std::endl;

  TransitionList steps;
  steps.push_back( TransitionRecord( "Processing", false ) );
  steps.push_back( TransitionRecord( "Running", false ) );
  steps.push_back( TransitionRecord( "Halting", true ) );
  steps.push_back( TransitionRecord( "Halting", false ) );
  steps.push_back( TransitionRecord( "Enabled", false ) );
  steps.push_back( TransitionRecord( "Ready", false ) );
  steps.push_back( TransitionRecord( "Halted", true ) );
  CPPUNIT_ASSERT( checkHistory( steps ) );

  machine_->terminate();
}

void testStateMachine::testReconfigureSequence()
{
  stor::EventPtr_t stMachEvent;

  resetStateMachine();
  sr_->statisticsReporter_->
    getStateMachineMonitorCollection().reset(boost::posix_time::not_a_date_time);
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  sr_->statisticsReporter_->
    getStateMachineMonitorCollection().reset(boost::posix_time::not_a_date_time);
  stMachEvent.reset( new Reconfigure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  std::cout << std::endl << "**** Testing if Reconfigure triggers the right sequence ****" << std::endl;

  TransitionList steps;
  steps.push_back( TransitionRecord( "Stopped", false ) );
  steps.push_back( TransitionRecord( "Ready", false ) );
  steps.push_back( TransitionRecord( "Ready", true ) );
  steps.push_back( TransitionRecord( "Stopped", true ) );
  CPPUNIT_ASSERT( checkHistory( steps ) );

  machine_->terminate();
}


void testStateMachine::testEmergencyStopSequence()
{
  stor::EventPtr_t stMachEvent;

  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  stMachEvent.reset( new Enable() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Processing" ) );

  sr_->statisticsReporter_->
    getStateMachineMonitorCollection().reset(boost::posix_time::not_a_date_time);
  stMachEvent.reset( new EmergencyStop() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  std::cout << std::endl << "**** Testing if EmergencyStop triggers the right sequence ****" << std::endl;

  TransitionList steps;
  steps.push_back( TransitionRecord( "Processing", false ) );
  steps.push_back( TransitionRecord( "Running", false ) );
  steps.push_back( TransitionRecord( "Stopping", true ) );
  steps.push_back( TransitionRecord( "Stopping", false ) );
  steps.push_back( TransitionRecord( "Enabled", false ) );
  steps.push_back( TransitionRecord( "Stopped", true ) );
  CPPUNIT_ASSERT( checkHistory( steps ) );

  machine_->terminate();
}

void testStateMachine::testAllStatesGoToFailed()
{
  stor::EventPtr_t stMachEvent;

  std::cout << std::endl << "**** Making sure Constructed can go to Failed ****" << std::endl;

  // Constructed:
  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Fail() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Failed" ) );
  machine_->terminate();

  std::cout << std::endl << "**** Making sure Halted can go to Failed ****" << std::endl;

  // Halted:
  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Fail() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Failed" ) );
  machine_->terminate();

  std::cout << std::endl << "**** Making sure Stopped can go to Failed ****" << std::endl;

  // Stopped:
  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  stMachEvent.reset( new Fail() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Failed" ) );
  machine_->terminate();
 
  std::cout << std::endl << "**** Making sure Processing can go to Failed ****" << std::endl;

  // Processing:
  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  stMachEvent.reset( new Enable() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Processing" ) );

  stMachEvent.reset( new Fail() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Failed" ) );
 machine_->terminate();

}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testStateMachine);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
