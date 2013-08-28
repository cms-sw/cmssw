#include <iostream>
#include <limits>

#include "FWCore/Utilities/interface/math.h"
#include "FWCore/Utilities/interface/HRRealTime.h"

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>


class TestMath : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestMath);
  CPPUNIT_TEST(test_isnan);
  CPPUNIT_TEST(timing);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown() {}
  void test_isnan();
  void timing();
};

template <class FP>
void
test_fp_type()
{
  CPPUNIT_ASSERT(std::numeric_limits<FP>::is_specialized);
  CPPUNIT_ASSERT(std::numeric_limits<FP>::has_quiet_NaN);
  FP nan = std::numeric_limits<FP>::quiet_NaN();
  CPPUNIT_ASSERT(!edm::isnan(static_cast<FP>(1.0)));
  CPPUNIT_ASSERT(!edm::isnan(static_cast<FP>(-1.0)));
  CPPUNIT_ASSERT(!edm::isnan(static_cast<FP>(0.0)));
  CPPUNIT_ASSERT(!edm::isnan(static_cast<FP>(-0.0)));
  CPPUNIT_ASSERT(!edm::isnan(std::numeric_limits<FP>::infinity()));
  CPPUNIT_ASSERT(!edm::isnan(-std::numeric_limits<FP>::infinity()));

  CPPUNIT_ASSERT(edm::isnan(nan));
  CPPUNIT_ASSERT(edm::isnan(std::numeric_limits<FP>::signaling_NaN()));
}



CPPUNIT_TEST_SUITE_REGISTRATION(TestMath);

void TestMath::test_isnan()
{
  test_fp_type<float>();
  test_fp_type<double>();
  test_fp_type<long double>();

  CPPUNIT_ASSERT(edm::isnan(std::numeric_limits<float>::quiet_NaN()));
  CPPUNIT_ASSERT(edm::isnan(std::numeric_limits<double>::quiet_NaN()));
  CPPUNIT_ASSERT(edm::isnan(std::numeric_limits<long double>::quiet_NaN()));
}

template <class FP>
void
time_fp_type()
{
  volatile float const zero = 0.0f; // Use a variable to avoid compiler warning on divide by zero.
  volatile FP values[] = { 1.0f, 1.0f/zero, -2.5f, 1.0f/3.0f, 0.0f/zero };

  unsigned long sum = 0;
  
  edm::HRTimeType start = edm::hrRealTime();
  for (int i = 0; i < 1000*1000; ++i)
    for (int j = 0; j < 5; ++j)
      sum += (std::isnan(values[j]) ? 0 : 1);
  edm::HRTimeType stop = edm::hrRealTime();
  std::cout << "std::isnan time:         " << (stop - start) << std::endl;

  sum = 0;
  start = edm::hrRealTime();
  for (int i = 0; i < 1000*1000; ++i)
    for (int j = 0; j < 5; ++j)
      sum += (edm::detail::isnan(values[j]) ? 0 : 1);
  stop = edm::hrRealTime();
  stop = edm::hrRealTime();
  std::cout << "edm::detail::isnan time: " << (stop - start) << std::endl;

  sum = 0;
  start = edm::hrRealTime();
  for (int i = 0; i < 1000*1000; ++i)
    for (int j = 0; j < 5; ++j)
      sum += (edm::isnan(values[j]) ? 0 : 1);
  stop = edm::hrRealTime();
  std::cout << "edm::isnan time:         " << (stop - start) << std::endl;

  sum = 0;
  start = edm::hrRealTime();
  for (int i = 0; i < 1000*1000; ++i)
    for (int j = 0; j < 5; ++j)
      sum += (edm::equal_isnan(values[j]) ? 0 : 1);
  stop = edm::hrRealTime();
  std::cout << "edm::equal_isnan time:   " << (stop - start) << std::endl;
}


void TestMath::timing()
{
  std::cout << "\n\ntiming floats\n";
  time_fp_type<float>();
  std::cout << "\ntiming doubles\n";
  time_fp_type<double>();
  std::cout << "\ntiming long doubles\n";
  time_fp_type<long double>();
}
