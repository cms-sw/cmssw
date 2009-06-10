#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/Utils.h"
#include "FWCore/Utilities/interface/CPUTimer.h"

void test_helper(edm::CPUTimer& t, 
                 stor::utils::duration_t interval, 
                 int expected_status,
                 double min_sleep, 
                 double max_sleep)
{
  t.start();
  int status = stor::utils::sleep(interval);
  t.stop();
  CPPUNIT_ASSERT(status == expected_status);
  CPPUNIT_ASSERT(t.realTime() >= min_sleep);
  CPPUNIT_ASSERT(t.realTime() <= max_sleep);  
}


class testSleep : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testSleep);

  CPPUNIT_TEST(negative_sleep_duration);
  CPPUNIT_TEST(zero_sleep_duration);
  CPPUNIT_TEST(subsecond_sleep_duration);
  CPPUNIT_TEST(multisecond_sleep_duration);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();

  void negative_sleep_duration();
  void zero_sleep_duration();
  void subsecond_sleep_duration();
  void multisecond_sleep_duration();

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
  test_helper(_timer, -0.1, -1, 0.0, _resolution);
}

void
testSleep::zero_sleep_duration()
{
  test_helper(_timer, 0.0, 0, 0.0, _max_allowed_shortest_sleep_duration);
}

void
testSleep::subsecond_sleep_duration()
{
  stor::utils::duration_t dur = 0.01;
  test_helper(_timer, dur, 0, 0.0, dur+_resolution);
}

void
testSleep::multisecond_sleep_duration()
{
  stor::utils::duration_t dur = 3.9;
  test_helper(_timer, dur, 0, 0.0, dur+_resolution);
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testSleep);

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
