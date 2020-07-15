// -*- C++ -*-
//
// Package:     PhysicsTools/MVAComputer
// Class  :     testBitSet.cppunit
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 23 Jan 2015 18:54:27 GMT
//

// system include files

// user include files
#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/MVAComputer/interface/BitSet.h"

class testBitSet : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testBitSet);

  CPPUNIT_TEST(bitManipulationTest);
  CPPUNIT_TEST(multiWordTest);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}

  void multiWordTest();
  void bitManipulationTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testBitSet);

using namespace PhysicsTools;

namespace {
  unsigned int bit(int pos) { return (~(1U << pos)) + 1; }

  unsigned int neg(int pos) { return -(1 << pos); }
}  // namespace

void testBitSet::bitManipulationTest() {
  for (unsigned int i = 0; i < 31; ++i) {
    CPPUNIT_ASSERT(bit(i) == neg(i));
  }

  //bit 31 was causing UBSAN problems for neg
  // so if neg != bit it is from the undefined
  // behavior
  CPPUNIT_ASSERT(bit(31) == 0x80000000);
  CPPUNIT_ASSERT(neg(31) == 0x80000000);
}

void testBitSet::multiWordTest() {
  BitSet b33(33);

  CPPUNIT_ASSERT(b33.size() == 33);
  CPPUNIT_ASSERT(b33.bits() == 0);

  CPPUNIT_ASSERT(bool(b33.iter()) == false);

  for (int i = 0; i < 33; ++i) {
    CPPUNIT_ASSERT(b33[i] == false);
  }

  b33[1] = true;
  CPPUNIT_ASSERT(b33[1] == true);
  CPPUNIT_ASSERT(b33.bits() == 1);

  {
    auto it = b33.iter();
    CPPUNIT_ASSERT(bool(it) == true);
    CPPUNIT_ASSERT(it() == 1);

    ++it;
    CPPUNIT_ASSERT(bool(it) == false);
  }

  b33[30] = true;
  CPPUNIT_ASSERT(b33.bits() == 2);
  {
    auto it = b33.iter();
    CPPUNIT_ASSERT(bool(it) == true);
    CPPUNIT_ASSERT(it() == 1);

    ++it;
    CPPUNIT_ASSERT(bool(it) == true);
    CPPUNIT_ASSERT(it() == 30);

    ++it;
    CPPUNIT_ASSERT(bool(it) == false);
  }

  b33[32] = true;
  CPPUNIT_ASSERT(b33.bits() == 3);
  {
    auto it = b33.iter();
    CPPUNIT_ASSERT(bool(it) == true);
    CPPUNIT_ASSERT(it() == 1);

    ++it;
    CPPUNIT_ASSERT(bool(it) == true);
    CPPUNIT_ASSERT(it() == 30);

    ++it;
    CPPUNIT_ASSERT(bool(it) == true);
    CPPUNIT_ASSERT(it() == 32);

    ++it;
    CPPUNIT_ASSERT(bool(it) == false);
  }
}
