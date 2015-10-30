#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include <iomanip>
#include <limits>

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
  int8_t pack(double x)          { return logintpack::pack8log        (x, -15, 0); }
  int8_t packceil(double x)      { return logintpack::pack8logCeil    (x, -15, 0); }
  int8_t packclosed(double x)    { return logintpack::pack8log        (x, -15, 0); }
  double unpack(int8_t x)        { return logintpack::unpack8log      (x, -15, 0); }
  double unpackclosed(int8_t x)  { return logintpack::unpack8logClosed(x, -15, 0); }
}

void testlogintpack::test() {
  using logintpack::smallestPositive;
  using logintpack::smallestNegative;
  constexpr int8_t largestPositive = 127;
  constexpr int8_t largestNegative = -127;

  const double smallestValuePos = std::exp(-15.);
  const double smallestValueNeg = -std::exp(-15.+1./128.*15.);
  const double smallestValueNegForClosed = -std::exp(-15.+1./127.*15.);
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

  const double largestValuePos = std::exp(-15.+127./128.*15.);
  const double largestValueNeg = -largestValuePos;
  CPPUNIT_ASSERT(pack(largestValuePos) == largestPositive);
  CPPUNIT_ASSERT(packceil(largestValuePos) == largestPositive);
  CPPUNIT_ASSERT(unpack(largestPositive) == largestValuePos);

  CPPUNIT_ASSERT(pack(largestValueNeg) == largestNegative);
  CPPUNIT_ASSERT(packceil(largestValueNeg) == largestNegative);
  CPPUNIT_ASSERT(unpack(largestNegative) == largestValueNeg);

  const double largestValueClosedPos = std::exp(0.);
  const double largestValueClosedNeg = -largestValueClosedPos;
  CPPUNIT_ASSERT(packclosed(largestValueClosedPos) == largestPositive);
  CPPUNIT_ASSERT(unpackclosed(largestPositive) == largestValueClosedPos);
  CPPUNIT_ASSERT(packclosed(largestValueClosedNeg) == largestNegative);
  CPPUNIT_ASSERT(unpackclosed(largestNegative) == largestValueClosedNeg);

  const double someValue = std::exp(-15. + 1/128.*15.);
  const float someValueFloat = std::exp(-15.f + 1/128.f*15.f);
  CPPUNIT_ASSERT(unpack(packceil(someValue)) == someValue);
  CPPUNIT_ASSERT(static_cast<float>(unpack(packceil(someValue))) == someValueFloat);
  {
    union { float flt; uint32_t i32; } conv;
    conv.flt = someValueFloat;
    conv.i32 += 1;
    const float someValuePlus1Ulp32 = conv.flt;
    CPPUNIT_ASSERT(static_cast<float>(unpack(packceil(someValuePlus1Ulp32))) >= someValuePlus1Ulp32);
  }
  {
    union { double flt; uint64_t i64; } conv;
    conv.flt = someValue;
    conv.i64 += 1;
    const float someValuePlus1Ulp64 = conv.flt;
    CPPUNIT_ASSERT(unpack(packceil(someValuePlus1Ulp64)) >= someValuePlus1Ulp64);
  }

}

CPPUNIT_TEST_SUITE_REGISTRATION(testlogintpack);

