#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include <iomanip>
#include <limits>

#include "DataFormats/Math/interface/liblogintpack.h"

class testlogintpack : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testlogintpack);

  CPPUNIT_TEST(test16base11);
  CPPUNIT_TEST(test8);

  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void test16base11();
  void test8();

private:
};

namespace {
  int16_t pack16base11(double x)          { return logintpack::pack16log        (x, -15, 0, 1<<11); }
  int16_t pack16base11ceil(double x)      { return logintpack::pack16logCeil    (x, -15, 0, 1<<11); }
  int16_t pack16base11closed(double x)    { return logintpack::pack16logClosed  (x, -15, 0, 1<<11); }
  double  unpack16base11(int16_t x)       { return logintpack::unpack16log      (x, -15, 0, 1<<11); }
  double  unpack16base11closed(int16_t x) { return logintpack::unpack16logClosed(x, -15, 0, 1<<11); }

  int8_t pack(double x)          { return logintpack::pack8log        (x, -15, 0); }
  int8_t packceil(double x)      { return logintpack::pack8logCeil    (x, -15, 0); }
  int8_t packclosed(double x)    { return logintpack::pack8log        (x, -15, 0); }
  double unpack(int8_t x)        { return logintpack::unpack8log      (x, -15, 0); }
  double unpackclosed(int8_t x)  { return logintpack::unpack8logClosed(x, -15, 0); }
}

void testlogintpack::test16base11() {
  constexpr uint16_t base = 1<<11; // 2^11 = 2048

  constexpr int16_t smallestPositive = 0;
  constexpr int16_t smallestNegative = -1;
  constexpr int16_t largestPositive = base-1;
  constexpr int16_t largestNegative = -largestPositive;

  union { float flt; uint32_t i32; } conv32;
  union { double flt; uint64_t i64; } conv64;

  const double smallestValuePos = std::exp(-15.);
  const double smallestValueNeg = -std::exp(-15. + 1./base*15.);
  const double smallestValueNegForClosed = -std::exp(-15. + 1./(base-1)*15.);
  CPPUNIT_ASSERT(pack16base11(smallestValuePos) == smallestPositive);
  CPPUNIT_ASSERT(pack16base11ceil(smallestValuePos) == smallestPositive);
  CPPUNIT_ASSERT(pack16base11closed(smallestValuePos) == smallestPositive);
  CPPUNIT_ASSERT(unpack16base11(smallestPositive) == smallestValuePos);
  CPPUNIT_ASSERT(unpack16base11closed(smallestPositive) == smallestValuePos);

  CPPUNIT_ASSERT(pack16base11(smallestValueNeg) == smallestNegative);
  CPPUNIT_ASSERT(pack16base11ceil(smallestValueNeg) == smallestNegative);
  CPPUNIT_ASSERT(unpack16base11(smallestNegative) == smallestValueNeg);
  CPPUNIT_ASSERT(unpack16base11(pack16base11(smallestValueNeg)) == smallestValueNeg);
  CPPUNIT_ASSERT(unpack16base11(pack16base11ceil(smallestValueNeg)) == smallestValueNeg);
  CPPUNIT_ASSERT(unpack16base11closed(pack16base11closed(smallestValueNegForClosed)) == smallestValueNegForClosed);

  CPPUNIT_ASSERT(pack16base11(smallestValuePos-std::exp(-16.)) == smallestPositive);
  CPPUNIT_ASSERT(pack16base11ceil(smallestValuePos-std::exp(-16.)) == smallestPositive);
  CPPUNIT_ASSERT(pack16base11closed(smallestValuePos-std::exp(-16.)) == smallestPositive);
  conv64.flt = smallestValuePos;
  conv64.i64 -= 1;
  const double smallestValuePosMinus1Ulp64 = conv64.flt;
  CPPUNIT_ASSERT(pack16base11(smallestValuePosMinus1Ulp64) == smallestPositive);
  CPPUNIT_ASSERT(pack16base11ceil(smallestValuePosMinus1Ulp64) == smallestPositive);
  CPPUNIT_ASSERT(pack16base11closed(smallestValuePosMinus1Ulp64) == smallestPositive);

  CPPUNIT_ASSERT(pack16base11(smallestValueNeg+std::exp(-16.)) == smallestNegative);
  CPPUNIT_ASSERT(pack16base11ceil(smallestValueNeg+std::exp(-16.)) == smallestNegative);
  CPPUNIT_ASSERT(unpack16base11(pack16base11(smallestValueNeg+std::exp(-16.))) == smallestValueNeg);
  CPPUNIT_ASSERT(unpack16base11(pack16base11ceil(smallestValueNeg+std::exp(-16.))) == smallestValueNeg);
  CPPUNIT_ASSERT(unpack16base11closed(pack16base11closed(smallestValueNegForClosed+std::exp(-16.))) == smallestValueNegForClosed);
  conv64.flt = smallestValueNeg;
  conv64.i64 += 1;
  const double smallestValueNegPlusUlp64 = conv64.flt;
  CPPUNIT_ASSERT(pack16base11(smallestValueNegPlusUlp64) == smallestNegative);
  CPPUNIT_ASSERT(pack16base11ceil(smallestValueNegPlusUlp64) == smallestNegative);
  CPPUNIT_ASSERT(unpack16base11(pack16base11(smallestValueNegPlusUlp64)) == smallestValueNeg);
  CPPUNIT_ASSERT(unpack16base11(pack16base11ceil(smallestValueNegPlusUlp64)) == smallestValueNeg);
  CPPUNIT_ASSERT(unpack16base11closed(pack16base11closed(smallestValueNegPlusUlp64)) == smallestValueNegForClosed);


  const double largestValuePos = std::exp(-15. + double(base-1)/base * 15.);
  const double largestValueNeg = -largestValuePos;
  CPPUNIT_ASSERT(pack16base11(largestValuePos) == largestPositive);
  CPPUNIT_ASSERT(pack16base11ceil(largestValuePos) == largestPositive);
  CPPUNIT_ASSERT(unpack16base11(largestPositive) == largestValuePos);

  CPPUNIT_ASSERT(pack16base11(largestValueNeg) == largestNegative);
  CPPUNIT_ASSERT(pack16base11ceil(largestValueNeg) == largestNegative);
  CPPUNIT_ASSERT(unpack16base11(largestNegative) == largestValueNeg);

  CPPUNIT_ASSERT(pack16base11(largestValuePos+0.1) == largestPositive);
  CPPUNIT_ASSERT(pack16base11ceil(largestValuePos+0.1) == largestPositive);

  CPPUNIT_ASSERT(pack16base11(largestValueNeg-0.1) == largestNegative);
  CPPUNIT_ASSERT(pack16base11ceil(largestValueNeg-0.1) == largestNegative);


  const double largestValueClosedPos = std::exp(0.);
  const double largestValueClosedNeg = -largestValueClosedPos;
  CPPUNIT_ASSERT(pack16base11closed(largestValueClosedPos) == largestPositive);
  CPPUNIT_ASSERT(unpack16base11closed(largestPositive) == largestValueClosedPos);
  CPPUNIT_ASSERT(pack16base11closed(largestValueClosedNeg) == largestNegative);
  CPPUNIT_ASSERT(unpack16base11closed(largestNegative) == largestValueClosedNeg);

  CPPUNIT_ASSERT(pack16base11closed(largestValueClosedPos+0.1) == largestPositive);
  CPPUNIT_ASSERT(pack16base11closed(largestValueClosedNeg-0.1) == largestNegative);

  const double someValue = std::exp(-15. + 1./base*15.);
  const float someValueFloat = std::exp(-15.f + 1.f/float(base)*15.f);
  CPPUNIT_ASSERT(unpack16base11(pack16base11ceil(someValue)) == someValue);
  CPPUNIT_ASSERT(static_cast<float>(unpack16base11(pack16base11ceil(someValue))) == someValueFloat);

  conv32.flt = someValueFloat;
  conv32.i32 += 1;
  const float someValuePlus1Ulp32 = conv32.flt;
  CPPUNIT_ASSERT(static_cast<float>(unpack16base11(pack16base11ceil(someValuePlus1Ulp32))) >= someValuePlus1Ulp32);

  conv64.flt = someValue;
  conv64.i64 += 1;
  const float someValuePlus1Ulp64 = conv64.flt;
  CPPUNIT_ASSERT(unpack16base11(pack16base11ceil(someValuePlus1Ulp64)) >= someValuePlus1Ulp64);
}

void testlogintpack::test8() {
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

