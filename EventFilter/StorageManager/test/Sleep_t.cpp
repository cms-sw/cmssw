#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/Utils.h"
#include "FWCore/Utilities/interface/CPUTimer.h"

void test_helper(edm::CPUTimer& t, 
                 stor::utils::duration_t interval, 
                 double min_sleep, 
                 double max_sleep)
{
  t.start();
  int status = stor::utils::sleep(interval);
  t.stop();
  if (interval < 0)
    CPPUNIT_ASSERT(status == -1);
  else
    CPPUNIT_ASSERT(status == 0);
  CPPUNIT_ASSERT(t.realTime() >= min_sleep);
  // maximum sleep time depends too much on the
  // system load to be meaningful
  //CPPUNIT_ASSERT(t.realTime() <= max_sleep);  
}

void test_helper_sleep_until(edm::CPUTimer& t, 
                             stor::utils::duration_t interval) 
{
  stor::utils::time_point_t now = stor::utils::getCurrentTime();
  int status = stor::utils::sleepUntil(now + interval);
  if (interval > 0)
    CPPUNIT_ASSERT(status == 0);
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
  edm::CPUTimer _timer;
  double        _resolution;
  double        _max_allowed_shortest_sleep_duration;
};


void
testSleep::setUp()
{
  _timer.start();
  _timer.stop();  
  _resolution = 2.0 * _timer.realTime();
  _timer.reset();
  _max_allowed_shortest_sleep_duration = 0.001; // 1 millisecond
}

void
testSleep::tearDown()
{ 
}


void 
testSleep::negative_sleep_duration()
{
  test_helper(_timer, -0.1, 0.0, _resolution);
}

void
testSleep::zero_sleep_duration()
{
  test_helper(_timer, 0.0, 0.0, _max_allowed_shortest_sleep_duration);
}

void
testSleep::subsecond_sleep_duration()
{
  stor::utils::duration_t dur = 0.01;
  test_helper(_timer, dur, 0.0, dur+_resolution);
}

void
testSleep::multisecond_sleep_duration()
{
  stor::utils::duration_t dur = 3.9;
  test_helper(_timer, dur, 0.0, dur+_resolution);
}

void
testSleep::negative_sleep_until()
{
  test_helper_sleep_until(_timer, -0.02);
}

void
testSleep::zero_sleep_until()
{
  test_helper_sleep_until(_timer, 0);
}

void
testSleep::subsecond_sleep_until()
{
  stor::utils::duration_t dur = 0.06;
  test_helper_sleep_until(_timer, dur);
}

void
testSleep::multisecond_sleep_until()
{
  stor::utils::duration_t dur = 3.1;
  test_helper_sleep_until(_timer, dur);
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testSleep);

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
