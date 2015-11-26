#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include <iomanip>

#include "DataFormats/PatCandidates/interface/liblogintpack.h"

class testlogintpack : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testlogintpack);

  CPPUNIT_TEST(test);

  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void test();

private:
};

namespace {
  double pack(double x)         { return logintpack::pack8log        (x, -15, 0); }
  double packceil(double x)     { return logintpack::pack8logCeil    (x, -15, 0); }
  double packclosed(double x)   { return logintpack::pack8log        (x, -15, 0); }
  double unpack(int8_t x)       { return logintpack::unpack8log      (x, -15, 0); }
  double unpackclosed(int8_t x) { return logintpack::unpack8logClosed(x, -15, 0); }
}

void testlogintpack::test() {
  using logintpack::smallestPositive;
  using logintpack::smallestNegative;
  constexpr int8_t largestPositive = 127;
  constexpr int8_t largestNegative = -127;

  const float smallestValuePos = std::exp(-15.f);
  const float smallestValueNeg = -std::exp(-15.f+1.f/128.f*15.f);
  const float smallestValueNegForClosed = -std::exp(-15.f+1.f/127.f*15.f);
  CPPUNIT_ASSERT(pack(smallestValuePos) == smallestPositive);
  CPPUNIT_ASSERT(packceil(smallestValuePos) == smallestPositive);
  CPPUNIT_ASSERT(packclosed(smallestValuePos) == smallestPositive);
  CPPUNIT_ASSERT(unpack(smallestPositive) == smallestValuePos);
  CPPUNIT_ASSERT(unpackclosed(smallestPositive) == smallestValuePos);

  CPPUNIT_ASSERT(pack(smallestValueNeg) == smallestNegative);
  CPPUNIT_ASSERT(packceil(smallestValueNeg) == smallestNegative);
  CPPUNIT_ASSERT(unpack(smallestNegative) == smallestValueNeg);
  CPPUNIT_ASSERT(unpack(pack(smallestValueNeg)) == smallestValueNeg);
  CPPUNIT_ASSERT(unpack(packceil(smallestValueNeg)) == smallestValueNeg);
  CPPUNIT_ASSERT(unpackclosed(packclosed(smallestValueNegForClosed)) == smallestValueNegForClosed);

  const float largestValuePos = std::exp(-15.f+127.f/128.f*15.f);
  const float largestValueNeg = -largestValuePos;
  CPPUNIT_ASSERT(pack(std::exp(0.f)) == largestPositive); // this one actually overflows
  CPPUNIT_ASSERT(pack(largestValuePos) == largestPositive);
  CPPUNIT_ASSERT(packceil(largestValuePos) == largestPositive);
  CPPUNIT_ASSERT(unpack(largestPositive) == largestValuePos);

  CPPUNIT_ASSERT(pack(largestValueNeg) == largestNegative);
  CPPUNIT_ASSERT(packceil(largestValueNeg) == largestNegative);
  CPPUNIT_ASSERT(unpack(largestNegative) == largestValueNeg);

  const float largestValueClosedPos = std::exp(0.f);
  const float largestValueClosedNeg = -largestValueClosedPos;
  CPPUNIT_ASSERT(packclosed(largestValueClosedPos) == largestPositive);
  CPPUNIT_ASSERT(unpackclosed(largestPositive) == largestValueClosedPos);
  CPPUNIT_ASSERT(packclosed(largestValueClosedNeg) == largestNegative);
  CPPUNIT_ASSERT(unpackclosed(largestNegative) == largestValueClosedNeg);
}

CPPUNIT_TEST_SUITE_REGISTRATION(testlogintpack);

