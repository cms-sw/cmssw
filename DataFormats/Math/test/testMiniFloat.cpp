#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

#include "DataFormats/Math/interface/libminifloat.h"
#include "FWCore/Utilities/interface/isFinite.h"

class testMiniFloat : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testMiniFloat);

  CPPUNIT_TEST(testIsDenorm);
  CPPUNIT_TEST(testMax);
  CPPUNIT_TEST(testMax32RoundedToMax16);
  CPPUNIT_TEST(testMin);
  CPPUNIT_TEST(testMin32RoundedToMin16);
  CPPUNIT_TEST(testDenormMin);

  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void testIsDenorm();
  void testMax() ;
  void testMax32RoundedToMax16();
  void testMin();
  void testMin32RoundedToMin16();
  void testDenormMin();

private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(testMiniFloat);

void testMiniFloat::testIsDenorm() {
  // all float16s with zero exponent and non-zero mantissa are denormals, test here the boundaries
  CPPUNIT_ASSERT(MiniFloatConverter::isdenorm(1));
  CPPUNIT_ASSERT(MiniFloatConverter::isdenorm(1 | (1<<15))); // negative 1
  CPPUNIT_ASSERT(MiniFloatConverter::isdenorm(0x3ff));
  CPPUNIT_ASSERT(MiniFloatConverter::isdenorm(0x3ff) | (1<<15)); // negative full-1 mantissa

  // Test also boundary cases for float16 not being denormal
  CPPUNIT_ASSERT(!MiniFloatConverter::isdenorm(0));
  CPPUNIT_ASSERT(!MiniFloatConverter::isdenorm(0x400)); // exponent 1, zero mantissa
  CPPUNIT_ASSERT(!MiniFloatConverter::isdenorm(0x400 | (1<<15))); // negative exponent 1, zero mantissa
}

void testMiniFloat::testMax() {
  // 0x1f exponent is for inf, so 0x1e is the maximum
  // in maximum mantissa all bits are 1
  const uint16_t minifloatmax = (0x1e << 10) | 0x3ff;
  CPPUNIT_ASSERT(MiniFloatConverter::max() == MiniFloatConverter::float16to32(minifloatmax));

  // adding 1ulp(16) to max should give inf
  const uint16_t minifloatinf = minifloatmax + 1;
  CPPUNIT_ASSERT(edm::isNotFinite(MiniFloatConverter::float16to32(minifloatinf)));
}

void testMiniFloat::testMax32RoundedToMax16() {
  // max32RoundedToMax16() -> float16 -> float32 should give max()
  CPPUNIT_ASSERT(MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(MiniFloatConverter::max32RoundedToMax16())) == MiniFloatConverter::max());

  // max32RoundedToMax16() + 1ulp(32) should give inf(16)
  union { float flt; uint32_t i32; } conv;
  conv.flt = MiniFloatConverter::max32RoundedToMax16();
  conv.i32 += 1;
  const float max32PlusUlp32RoundedTo16 = MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(conv.flt));
  CPPUNIT_ASSERT(edm::isNotFinite(max32PlusUlp32RoundedTo16));
}

void testMiniFloat::testMin() {
  // 1 exponent, and 0 mantissa gives the smallest non-denormalized number of float16
  CPPUNIT_ASSERT(MiniFloatConverter::min() == MiniFloatConverter::float16to32(1 << 10));

  // subtracting 1ulp(16) from min should give denormalized float16
  const uint16_t minifloat_denorm = MiniFloatConverter::float32to16(MiniFloatConverter::min()) - 1;
  CPPUNIT_ASSERT(MiniFloatConverter::isdenorm(minifloat_denorm));

  // subtracking 1ulp(32) from min should also give denormalized float16 (both crop and round)
  union { float flt; uint32_t i32; } conv;
  conv.flt = MiniFloatConverter::min();
  conv.i32 -= 1;
  const uint16_t min32MinusUlp32CroppedTo16 = MiniFloatConverter::float32to16crop(conv.flt);
  CPPUNIT_ASSERT(MiniFloatConverter::isdenorm(min32MinusUlp32CroppedTo16));
  const uint16_t min32MinusUlp32RoundedTo16 = MiniFloatConverter::float32to16round(conv.flt);
  CPPUNIT_ASSERT(MiniFloatConverter::isdenorm(min32MinusUlp32RoundedTo16));
}

void testMiniFloat::testMin32RoundedToMin16() {
  // min32RoundedToMin16() -> float16 -> float32 should be the same as min()
  CPPUNIT_ASSERT(MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(MiniFloatConverter::min32RoundedToMin16())) == MiniFloatConverter::min());

  // min32RoundedToMax16() - 1ulp(32) should give denormalized float16
  union { float flt; uint32_t i32; } conv;
  conv.flt = MiniFloatConverter::min32RoundedToMin16();
  conv.i32 -= 1;
  const uint16_t min32MinusUlp32RoundedTo16 = MiniFloatConverter::float32to16(conv.flt);
  CPPUNIT_ASSERT(MiniFloatConverter::isdenorm(min32MinusUlp32RoundedTo16));
}

void testMiniFloat::testDenormMin() {
  // zero exponent, and 0x1 in mantissa gives the smallest number of float16
  CPPUNIT_ASSERT(MiniFloatConverter::denorm_min() == MiniFloatConverter::float16to32(1));

  // subtracting 1ulp(16) from denorm_min should give 0 float32
  CPPUNIT_ASSERT(MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(MiniFloatConverter::denorm_min()) - 1) == 0.f);

  // subtracking 1ulp(32) from denorm_min should also give 0 float32
  union { float flt; uint32_t i32; } conv;
  conv.flt = MiniFloatConverter::denorm_min();
  conv.i32 -= 1;
  const float min32MinusUlp32RoundedTo16 = MiniFloatConverter::float16to32(MiniFloatConverter::float32to16round(conv.flt));
  CPPUNIT_ASSERT(min32MinusUlp32RoundedTo16 == 0.f);
  const float min32MinusUlp32CroppedTo16 = MiniFloatConverter::float16to32(MiniFloatConverter::float32to16crop(conv.flt));
  CPPUNIT_ASSERT(min32MinusUlp32CroppedTo16 == 0.f);
}
