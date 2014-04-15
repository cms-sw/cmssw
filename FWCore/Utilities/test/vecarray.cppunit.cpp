#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/Utilities/interface/VecArray.h"

class testVecArray: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testVecArray);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void test();
  template <typename T>
  void testIterators(T& array);
  template <typename T>
  void testIterator(T iter, T end);

};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testVecArray);

void testVecArray::test() {
  edm::VecArray<int, 4> array;
  CPPUNIT_ASSERT(array.empty());
  CPPUNIT_ASSERT(array.size() == 0);
  CPPUNIT_ASSERT(array.capacity() == 4);
  CPPUNIT_ASSERT((edm::VecArray<int, 4>::capacity() == 4));

  auto iter = array.begin();
  auto end = array.end();
  CPPUNIT_ASSERT(iter == end);

  array.push_back_unchecked(1);
  CPPUNIT_ASSERT(!array.empty());
  CPPUNIT_ASSERT(array.size() == 1);
  CPPUNIT_ASSERT(array.front() == 1);
  CPPUNIT_ASSERT(array.back() == 1);
  iter = array.begin();
  end = array.end();
  CPPUNIT_ASSERT(iter != end);
  CPPUNIT_ASSERT(*iter == 1);
  ++iter;
  CPPUNIT_ASSERT(iter == end);

  array.push_back_unchecked(2);
  CPPUNIT_ASSERT(array.size() == 2);
  CPPUNIT_ASSERT(array.back() == 2);
  array.push_back(3);
  CPPUNIT_ASSERT(array.size() == 3);
  CPPUNIT_ASSERT(array.back() == 3);
  array.push_back(4);
  CPPUNIT_ASSERT(array.size() == 4);
  CPPUNIT_ASSERT(array.front() == 1);
  CPPUNIT_ASSERT(array.back() == 4);
  CPPUNIT_ASSERT(array[0] == 1 && array[1] == 2 && array[2] == 3 && array[3] == 4);

  try {
    array.push_back(5);
    CPPUNIT_ASSERT(false);
  } catch(std::length_error& e) {
    CPPUNIT_ASSERT(true);
  }

  auto ptr = array.data();
  CPPUNIT_ASSERT(ptr[0] == 1 && ptr[1] == 2 && ptr[2] == 3 && ptr[3] == 4);

  testIterators(array);
  testIterators(const_cast<const decltype(array)&>(array));

  edm::VecArray<int, 4> array2;
  array2.push_back(11);
  array2.push_back(12);

  array.swap(array2);
  CPPUNIT_ASSERT(array.size() == 2);
  CPPUNIT_ASSERT(array2.size() == 4);
  CPPUNIT_ASSERT(array[0] == 11 && array[1] == 12);
  CPPUNIT_ASSERT(array2[0] == 1 && array2[1] == 2 && array2[2] == 3 && array2[3] == 4);

  array = array2;
  CPPUNIT_ASSERT(array.size() == array2.size());
  CPPUNIT_ASSERT(array.size() == 4);
  CPPUNIT_ASSERT(array[0] == 1 && array[1] == 2 && array[2] == 3 && array[3] == 4);
  CPPUNIT_ASSERT(array2[0] == 1 && array2[1] == 2 && array2[2] == 3 && array2[3] == 4);

  ptr = array.data();
  ptr[1] = 10;
  iter = array.begin()+1;
  CPPUNIT_ASSERT(*iter == 10);

  iter = array.begin()+2;
  *iter = 50;
  CPPUNIT_ASSERT(array[0] == 1 && array[1] == 10 && array[2] == 50 && array[3] == 4);

  CPPUNIT_ASSERT(!array.empty());
  array.clear();
  CPPUNIT_ASSERT(array.empty());
  CPPUNIT_ASSERT(array.size() == 0);
  CPPUNIT_ASSERT(array2.size() == 4);

  array.resize(2);
  CPPUNIT_ASSERT(array.size() == 2);
  array.pop_back();
  CPPUNIT_ASSERT(array.size() == 1);
  try {
    array.resize(6);
    CPPUNIT_ASSERT(false);
  } catch(std::length_error& e) {
    CPPUNIT_ASSERT(true);
  }
  CPPUNIT_ASSERT(array.size() == 1);
  array.resize(4);
  CPPUNIT_ASSERT(array.size() == 4);
  array.resize(1);
  CPPUNIT_ASSERT(array.size() == 1);
}

template <typename T>
void testVecArray::testIterators(T& array) {
  testIterator(array.begin(), array.end());
  testIterator(array.cbegin(), array.cend());
}
template <typename T>
void testVecArray::testIterator(T iter, T end) {
  CPPUNIT_ASSERT(iter != end);
  CPPUNIT_ASSERT(*iter == 1);
  ++iter;
  CPPUNIT_ASSERT(iter != end);
  CPPUNIT_ASSERT(*iter == 2);
  ++iter;
  CPPUNIT_ASSERT(iter != end);
  CPPUNIT_ASSERT(*iter == 3);
  ++iter;
  CPPUNIT_ASSERT(iter != end);
  CPPUNIT_ASSERT(*iter == 4);
  ++iter;
  CPPUNIT_ASSERT(iter == end);
}
