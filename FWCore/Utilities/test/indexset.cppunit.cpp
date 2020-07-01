#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/Utilities/interface/IndexSet.h"

class testIndexSet : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testIndexSet);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}

  void test();
  template <typename T>
  void testIterators(T& array);
  template <typename T>
  void testIterator(T iter, T end);
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testIndexSet);

void testIndexSet::test() {
  edm::IndexSet set;
  CPPUNIT_ASSERT(set.empty());
  CPPUNIT_ASSERT(set.empty());
  CPPUNIT_ASSERT(!set.has(0));

  set.reserve(10);

  CPPUNIT_ASSERT(set.empty());
  CPPUNIT_ASSERT(set.empty());
  CPPUNIT_ASSERT(!set.has(0));

  set.insert(0);
  CPPUNIT_ASSERT(!set.empty());
  CPPUNIT_ASSERT(set.size() == 1);
  CPPUNIT_ASSERT(set.has(0));
  CPPUNIT_ASSERT(!set.has(1));

  set.insert(2);
  CPPUNIT_ASSERT(set.size() == 2);
  CPPUNIT_ASSERT(set.has(0));
  CPPUNIT_ASSERT(!set.has(1));
  CPPUNIT_ASSERT(set.has(2));
  CPPUNIT_ASSERT(!set.has(3));

  set.insert(20);
  CPPUNIT_ASSERT(set.size() == 3);
  CPPUNIT_ASSERT(set.has(0));
  CPPUNIT_ASSERT(!set.has(1));
  CPPUNIT_ASSERT(set.has(2));
  CPPUNIT_ASSERT(!set.has(3));
  CPPUNIT_ASSERT(!set.has(19));
  CPPUNIT_ASSERT(set.has(20));
  CPPUNIT_ASSERT(!set.has(21));

  set.insert(2);
  CPPUNIT_ASSERT(set.size() == 3);
  CPPUNIT_ASSERT(set.has(2));

  set.clear();
  CPPUNIT_ASSERT(set.empty());
  CPPUNIT_ASSERT(set.empty());
  CPPUNIT_ASSERT(!set.has(0));
  CPPUNIT_ASSERT(!set.has(1));
  CPPUNIT_ASSERT(!set.has(2));
  CPPUNIT_ASSERT(!set.has(3));
}
