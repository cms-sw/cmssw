#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "boost/date_time/posix_time/posix_time_types.hpp"

#include "EventFilter/StorageManager/interface/CommandQueue.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"


class testCommandQueue : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testCommandQueue);
  CPPUNIT_TEST(default_q_is_empty);
  CPPUNIT_TEST(matched_deqs_and_enqs);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();

  void default_q_is_empty();
  void matched_deqs_and_enqs();

private:
  // No data members yet.
};

void
testCommandQueue::setUp()
{ 
}

void
testCommandQueue::tearDown()
{ 
}


void 
testCommandQueue::default_q_is_empty()
{
  stor::CommandQueue q;
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(q.size() ==0);  
}

void
testCommandQueue::matched_deqs_and_enqs()
{
  unsigned long wait_usec = 1000; // wait time in microseconds
  typedef boost::shared_ptr<boost::statechart::event_base> event_ptr;
  stor::CommandQueue q;
  q.enqTimedWait(event_ptr(new stor::Configure), boost::posix_time::microseconds(wait_usec));
  CPPUNIT_ASSERT(q.size() == 1);
  q.enqTimedWait(event_ptr(new stor::Enable), boost::posix_time::microseconds(wait_usec));
  q.enqTimedWait(event_ptr(new stor::EmergencyStop), boost::posix_time::microseconds(wait_usec));
  CPPUNIT_ASSERT(q.size() == 3);
  event_ptr discard;
  CPPUNIT_ASSERT(q.deqNowait(discard));
  CPPUNIT_ASSERT(q.deqNowait(discard));
  CPPUNIT_ASSERT(q.deqNowait(discard));
  CPPUNIT_ASSERT(q.empty());
}


// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testCommandQueue);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
