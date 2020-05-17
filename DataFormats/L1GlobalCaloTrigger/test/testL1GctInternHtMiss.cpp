// Unit test for L1GctInternHtMiss class.
//
// Author Robert Frazier

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternHtMiss.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <cstdlib>

using namespace std;

uint16_t gCapBlockVal = 0x582;
uint16_t gCapIndexVal = 3;
int16_t gBxVal = 55;
uint32_t gRawVal1 = 0xf9ef8ecf;
uint32_t gRawVal2 = 0x394e1d9f;

// Prototypes
void terminateIfAssertNotWorking();
void testMethodsOfNullVariant();
void testMethodsOfHtxVariant();
void testMethodsOfHtyVariant();
void testMethodsOfHtxHtyVariant();
void testInequalityOperatorsBetweenCtors();

int main() {
  terminateIfAssertNotWorking();

  testMethodsOfNullVariant();
  testMethodsOfHtxVariant();
  testMethodsOfHtyVariant();
  testMethodsOfHtxHtyVariant();
  testInequalityOperatorsBetweenCtors();

  cout << "Unit test for L1GctInternHtMiss passed successfully." << endl;
  return 0;
}

void terminateIfAssertNotWorking() {
  bool assertsWork = false;
  assert(assertsWork = true);
  if (assertsWork) {
    return;
  }

  cout << "ERROR! Cannot run unit test as the assert() function is being\n"
          "optimised away by the compiler.  Please recompile test with\n"
          "the debug options enabled and without #define NDEBUG"
       << endl;
  exit(1);
}

void testMethodsOfNullVariant() {
  L1GctInternHtMiss testObj1;

  assert(testObj1.type() == L1GctInternHtMiss::nulltype);
  assert(testObj1.capBlock() == 0);
  assert(testObj1.capIndex() == 0);
  assert(testObj1.bx() == 0);
  assert(testObj1.isThereHtx() == false);
  assert(testObj1.isThereHty() == false);
  assert(testObj1.raw() == 0);  // The non-data in the raw data (BC0, etc) is masked off.
  assert(testObj1.htx() == 0);
  assert(testObj1.hty() == 0);
  assert(testObj1.overflow() == false);

  ostringstream oss;
  oss << testObj1;
  string expectedStr =
      " L1GctInternHtMiss:  htx=n/a, hty=n/a;"
      " cap block=0x0, index=0, BX=0";
  assert(oss.str() == expectedStr);
}

void testMethodsOfHtxVariant() {
  L1GctInternHtMiss testObj1 = L1GctInternHtMiss::unpackerMissHtx(gCapBlockVal, gCapIndexVal, gBxVal, gRawVal1);

  assert(testObj1.type() == L1GctInternHtMiss::miss_htx);
  assert(testObj1.capBlock() == gCapBlockVal);
  assert(testObj1.capIndex() == gCapIndexVal);
  assert(testObj1.bx() == gBxVal);
  assert(testObj1.isThereHtx() == true);
  assert(testObj1.isThereHty() == false);
  assert(testObj1.raw() == (gRawVal1 & 0x4000ffff));  // The non-data in the raw data (BC0, etc) is masked off.
  assert(testObj1.htx() == -28977);
  assert(testObj1.hty() == 0);
  assert(testObj1.overflow() == true);

  L1GctInternHtMiss testObj2 = L1GctInternHtMiss::unpackerMissHtx(gCapBlockVal, gCapIndexVal, gBxVal, gRawVal2);
  assert(testObj2.raw() == (gRawVal2 & 0x4000ffff));  // The non-data in the raw data (BC0, etc) is masked off.
  assert(testObj2.htx() == 7583);
  assert(testObj2.hty() == 0);
  assert(testObj2.overflow() == false);

  const L1GctInternHtMiss& testObj3(testObj1);  // Copy constructor

  assert(testObj1 == testObj3);  // Test equality & copy ctor.
  assert(testObj1 != testObj2);  // Test inequality.

  testObj2 = testObj3;  // Assignment operator

  assert(testObj1 == testObj2);  // Test Assignment operator + equality

  ostringstream oss;
  oss << testObj1;
  string expectedStr =
      " L1GctInternHtMiss:  htx=-28977, hty=n/a; overflow set;"
      " cap block=0x582, index=3, BX=55";
  assert(oss.str() == expectedStr);
}

void testMethodsOfHtyVariant() {
  L1GctInternHtMiss testObj1 = L1GctInternHtMiss::unpackerMissHty(gCapBlockVal, gCapIndexVal, gBxVal, gRawVal1);

  assert(testObj1.type() == L1GctInternHtMiss::miss_hty);
  assert(testObj1.capBlock() == gCapBlockVal);
  assert(testObj1.capIndex() == gCapIndexVal);
  assert(testObj1.bx() == gBxVal);
  assert(testObj1.isThereHtx() == false);
  assert(testObj1.isThereHty() == true);
  assert(testObj1.raw() == (gRawVal1 & 0x4000ffff));  // The non-data in the raw data (BC0, etc) is masked off.
  assert(testObj1.htx() == 0);
  assert(testObj1.hty() == -28977);
  assert(testObj1.overflow() == true);

  L1GctInternHtMiss testObj2 = L1GctInternHtMiss::unpackerMissHty(gCapBlockVal, gCapIndexVal, gBxVal, gRawVal2);
  assert(testObj2.raw() == (gRawVal2 & 0x4000ffff));  // The non-data in the raw data (BC0, etc) is masked off.
  assert(testObj2.htx() == 0);
  assert(testObj2.hty() == 7583);
  assert(testObj2.overflow() == false);

  const L1GctInternHtMiss& testObj3(testObj1);  // Copy constructor

  assert(testObj1 == testObj3);  // Test equality & copy ctor.
  assert(testObj1 != testObj2);  // Test inequality.

  testObj2 = testObj3;  // Assignment operator

  assert(testObj1 == testObj2);  // Test Assignment operator + equality

  ostringstream oss;
  oss << testObj1;
  string expectedStr =
      " L1GctInternHtMiss:  htx=n/a, hty=-28977; overflow set;"
      " cap block=0x582, index=3, BX=55";
  assert(oss.str() == expectedStr);
}

void testMethodsOfHtxHtyVariant() {
  L1GctInternHtMiss testObj1 = L1GctInternHtMiss::unpackerMissHtxHty(gCapBlockVal, gCapIndexVal, gBxVal, gRawVal1);

  assert(testObj1.type() == L1GctInternHtMiss::miss_htx_and_hty);
  assert(testObj1.capBlock() == gCapBlockVal);
  assert(testObj1.capIndex() == gCapIndexVal);
  assert(testObj1.bx() == gBxVal);
  assert(testObj1.isThereHtx() == true);
  assert(testObj1.isThereHty() == true);
  assert(testObj1.raw() == (gRawVal1 & 0x3fffbfff));  // The non-data in the raw data (BC0, etc) is masked off.
  assert(testObj1.htx() == 3791);
  assert(testObj1.hty() == -1553);
  assert(testObj1.overflow() == true);

  L1GctInternHtMiss testObj2 = L1GctInternHtMiss::unpackerMissHtxHty(gCapBlockVal, gCapIndexVal, gBxVal, gRawVal2);
  assert(testObj2.raw() == (gRawVal2 & 0x3fffbfff));  // The non-data in the raw data (BC0, etc) is masked off.
  assert(testObj2.htx() == 7583);
  assert(testObj2.hty() == -1714);
  assert(testObj2.overflow() == false);

  const L1GctInternHtMiss& testObj3(testObj1);  // Copy constructor

  assert(testObj1 == testObj3);  // Test equality & copy ctor.
  assert(testObj1 != testObj2);  // Test inequality.

  testObj2 = testObj3;  // Assignment operator

  assert(testObj1 == testObj2);  // Test Assignment operator + equality

  ostringstream oss;
  oss << testObj1;
  string expectedStr =
      " L1GctInternHtMiss:  htx=3791, hty=-1553; overflow set;"
      " cap block=0x582, index=3, BX=55";
  assert(oss.str() == expectedStr);
}

void testInequalityOperatorsBetweenCtors() {
  L1GctInternHtMiss testObj1 = L1GctInternHtMiss::unpackerMissHtx(gCapBlockVal, gCapIndexVal, gBxVal, gRawVal1);
  L1GctInternHtMiss testObj2 = L1GctInternHtMiss::unpackerMissHty(gCapBlockVal, gCapIndexVal, gBxVal, gRawVal1);
  L1GctInternHtMiss testObj3 = L1GctInternHtMiss::unpackerMissHtxHty(gCapBlockVal, gCapIndexVal, gBxVal, gRawVal1);

  assert(testObj1 != testObj2);
  assert(testObj1 != testObj3);
  assert(testObj2 != testObj3);
}
