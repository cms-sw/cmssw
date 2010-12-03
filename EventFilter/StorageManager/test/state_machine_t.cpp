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
  void processEvent( stor::event_ptr requestedEvent );
  bool checkState( const std::string& expected );
  void checkSignals( const EventList& elist, const std::string& expected);
  bool checkHistory( const TransitionList& steps );

  static xdaq::Application* _app;
  EventDistributor* _ed;
  FragmentStore* _fs;
  MockNotifier* _mn;

  StateMachine *_machine;
  static SharedResourcesPtr _sr;
};

xdaq::Application* testStateMachine::_app;
SharedResourcesPtr testStateMachine::_sr;

void testStateMachine::setUp()
{
  // 30-Jun-2009, KAB - to avoid the problem in which we try to re-declare infospace
  // variables to the MockApplication infospace for each test, we need to create just
  // one mock application and one shared resources instance for all tests.
  if ( _sr.get() == 0 )
  {
    mkdir("/tmp/log", 644); // dummy dir to avoid DiskSpaceAlarms
    _app = mockapps::getMockXdaqApplication();

    _sr.reset(new SharedResources());
    _sr->_configuration.reset(new Configuration(_app->getApplicationInfoSpace(), 0));
    _sr->_initMsgCollection.reset(new InitMsgCollection());
    _sr->_diskWriterResources.reset(new MockDiskWriterResources());
    _sr->_dqmEventProcessorResources.reset(new MockDQMEventProcessorResources());
    _sr->_commandQueue.reset(new CommandQueue(32));
    _sr->_fragmentQueue.reset(new FragmentQueue(32));
    _sr->_registrationQueue.reset(new RegistrationQueue(32));
    _sr->_streamQueue.reset(new StreamQueue(32));
    _sr->_dqmEventQueue.reset(new DQMEventQueue(32));
    _sr->_statisticsReporter.reset( new StatisticsReporter( _app, _sr ) );
    EventConsumerMonitorCollection& ecmc = 
      _sr->_statisticsReporter->getEventConsumerMonitorCollection();
    _sr->_eventConsumerQueueCollection.reset( new EventQueueCollection( ecmc ) );
    DQMConsumerMonitorCollection& dcmc = 
      _sr->_statisticsReporter->getDQMConsumerMonitorCollection();
    _sr->_dqmEventConsumerQueueCollection.reset( new DQMEventQueueCollection( dcmc ) );
    
    _sr->_discardManager.reset(new DiscardManager(_app->getApplicationContext(),
                                                  _app->getApplicationDescriptor(),
                                                  _sr->_statisticsReporter->getDataSenderMonitorCollection()));

    _sr->_registrationCollection.reset(new RegistrationCollection());
  }

  _ed = new EventDistributor(_sr);
  _fs = new FragmentStore();
  _mn = new MockNotifier( _app );

  _machine = new StateMachine( _ed, _fs, _mn, _sr );
}

void testStateMachine::tearDown()
{
  //delete _machine;
}

void testStateMachine::resetStateMachine()
{
  _sr->_statisticsReporter->
    getStateMachineMonitorCollection().reset(0);
  _machine->initiate();
}

/////////////////////////////////////////////////////////////////////
//// Simulate the processing of events by the fragment processor ////
/////////////////////////////////////////////////////////////////////
void testStateMachine::processEvent( stor::event_ptr requestedEvent )
{
  boost::shared_ptr<CommandQueue> cmdQueue;
  cmdQueue = _machine->getSharedResources()->_commandQueue;

  cmdQueue->enq_wait( requestedEvent );
  
  stor::event_ptr nextEvent;
  while ( cmdQueue->deq_nowait( nextEvent ) )
  {
    _machine->process_event( *nextEvent );
    _machine->getCurrentState().noFragmentToProcess();
  }
}


////////////////////////////////////////////
//// Returns false if unexpected state: ////
////////////////////////////////////////////
bool testStateMachine::checkState( const std::string& expected )
{
  const std::string actual = _machine->getCurrentStateName();
  if( actual != expected )
  {
    std::cerr << "Expecting " << expected << ", got " << actual << std::endl;
    _sr->_statisticsReporter->
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
  _sr->_statisticsReporter->
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

  _machine->terminate();
}


void testStateMachine::testHalted()
{
  stor::event_ptr stMachEvent;

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

  _machine->terminate();
}

void testStateMachine::testStopped()
{
  stor::event_ptr stMachEvent;

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

  _machine->terminate();
}

void testStateMachine::testProcessing()
{
  stor::event_ptr stMachEvent;

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

  _machine->terminate();
}


void testStateMachine::testFail()
{
  stor::event_ptr stMachEvent;

  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Fail() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Failed" ) );

  std::cout << std::endl << "**** Making sure no signal changes Failed state ****" << std::endl;

  EventList elist;
  checkSignals( elist, "Failed" );

  _machine->terminate();
}


void testStateMachine::testEnableSequence()
{
  stor::event_ptr stMachEvent;

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

  _machine->terminate();
}

void testStateMachine::testStopSequence()
{
  stor::event_ptr stMachEvent;

  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  stMachEvent.reset( new Enable() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Processing" ) );

  _sr->_statisticsReporter->
    getStateMachineMonitorCollection().reset(0);
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

  _machine->terminate();
}

void testStateMachine::testHaltSequence()
{
  stor::event_ptr stMachEvent;

  resetStateMachine();
  _sr->_statisticsReporter->
    getStateMachineMonitorCollection().reset(0);
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  stMachEvent.reset( new Enable() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Processing" ) );

  _sr->_statisticsReporter->
    getStateMachineMonitorCollection().reset(0);
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

  _machine->terminate();
}

void testStateMachine::testReconfigureSequence()
{
  stor::event_ptr stMachEvent;

  resetStateMachine();
  _sr->_statisticsReporter->
    getStateMachineMonitorCollection().reset(0);
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  _sr->_statisticsReporter->
    getStateMachineMonitorCollection().reset(0);
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

  _machine->terminate();
}


void testStateMachine::testEmergencyStopSequence()
{
  stor::event_ptr stMachEvent;

  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Configure() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Stopped" ) );

  stMachEvent.reset( new Enable() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Processing" ) );

  _sr->_statisticsReporter->
    getStateMachineMonitorCollection().reset(0);
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

  _machine->terminate();
}

void testStateMachine::testAllStatesGoToFailed()
{
  stor::event_ptr stMachEvent;

  std::cout << std::endl << "**** Making sure Constructed can go to Failed ****" << std::endl;

  // Constructed:
  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Fail() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Failed" ) );
  _machine->terminate();

  std::cout << std::endl << "**** Making sure Halted can go to Failed ****" << std::endl;

  // Halted:
  resetStateMachine();
  CPPUNIT_ASSERT( checkState( "Constructed" ) );

  stMachEvent.reset( new Fail() );
  processEvent( stMachEvent );
  CPPUNIT_ASSERT( checkState( "Failed" ) );
  _machine->terminate();

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
  _machine->terminate();
 
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
 _machine->terminate();

}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testStateMachine);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
