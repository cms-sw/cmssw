#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/Utils.h"
#include "FWCore/Utilities/interface/CPUTimer.h"

void test_helper(edm::CPUTimer& t, 
                 stor::utils::Duration_t interval, 
                 double min_sleep, 
                 double max_sleep)
{
  t.start();
  stor::utils::sleep(interval);
  t.stop();
  CPPUNIT_ASSERT(t.realTime() >= min_sleep);
  // maximum sleep time depends too much on the
  // system load to be meaningful
  //CPPUNIT_ASSERT(t.realTime() <= max_sleep);  
}

void test_helper_sleep_until(edm::CPUTimer& t, 
                             stor::utils::Duration_t interval) 
{
  stor::utils::TimePoint_t now = stor::utils::getCurrentTime();
  stor::utils::sleepUntil(now + interval);
  CPPUNIT_ASSERT(stor::utils::getCurrentTime() >= now+interval);
}

class testSleep : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testSleep);

  CPPUNIT_TEST(negative_sleep_duration);
  CPPUNIT_TEST(zero_sleep_duration);
  CPPUNIT_TEST(subsecond_sleep_duration);
  CPPUNIT_TEST(multisecond_sleep_duration);
  CPPUNIT_TEST(negative_sleep_until);
  CPPUNIT_TEST(zero_sleep_until);
  CPPUNIT_TEST(subsecond_sleep_until);
  CPPUNIT_TEST(multisecond_sleep_until);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();

  void negative_sleep_duration();
  void zero_sleep_duration();
  void subsecond_sleep_duration();
  void multisecond_sleep_duration();
  void negative_sleep_until();
  void zero_sleep_until();
  void subsecond_sleep_until();
  void multisecond_sleep_until();

private:
  edm::CPUTimer timer_;
  double        resolution_;
  double        max_allowed_shortest_sleep_duration_;
};


void
testSleep::setUp()
{
  timer_.start();
  timer_.stop();  
  resolution_ = 2.0 * timer_.realTime();
  timer_.reset();
  max_allowed_shortest_sleep_duration_ = 0.001; // 1 millisecond
}

void
testSleep::tearDown()
{ 
}


void 
testSleep::negative_sleep_duration()
{
  test_helper(timer_, boost::posix_time::time_duration(0,-1,0), 0.0, resolution_);
}

void
testSleep::zero_sleep_duration()
{
  test_helper(timer_, boost::posix_time::time_duration(0,0,0), 0.0, max_allowed_shortest_sleep_duration_);
}

void
testSleep::subsecond_sleep_duration()
{
  test_helper(timer_, boost::posix_time::time_duration(0,0,0,1000), 0.0, 0.1+resolution_);
}

void
testSleep::multisecond_sleep_duration()
{
  test_helper(timer_, boost::posix_time::time_duration(0,0,3,9000), 0.0, 3.9+resolution_);
}

void
testSleep::negative_sleep_until()
{
  test_helper_sleep_until(timer_, boost::posix_time::time_duration(0,-1,0));
}

void
testSleep::zero_sleep_until()
{
  test_helper_sleep_until(timer_, boost::posix_time::time_duration(0,0,0));
}

void
testSleep::subsecond_sleep_until()
{
  test_helper_sleep_until(timer_, boost::posix_time::time_duration(0,0,0,600));
}

void
testSleep::multisecond_sleep_until()
{
  test_helper_sleep_until(timer_, boost::posix_time::time_duration(0,0,3,1000));
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testSleep);

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
