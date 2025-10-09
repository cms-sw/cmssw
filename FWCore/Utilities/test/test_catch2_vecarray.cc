#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/VecArray.h"

template <typename T>
void testIterator(T iter, T end) {
  REQUIRE(iter != end);
  REQUIRE(*iter == 1);
  ++iter;
  REQUIRE(iter != end);
  REQUIRE(*iter == 2);
  ++iter;
  REQUIRE(iter != end);
  REQUIRE(*iter == 3);
  ++iter;
  REQUIRE(iter != end);
  REQUIRE(*iter == 4);
  ++iter;
  REQUIRE(iter == end);
}

template <typename T>
void testIterators(T& array) {
  testIterator(array.begin(), array.end());
  testIterator(array.cbegin(), array.cend());
}

TEST_CASE("edm::VecArray", "[VecArray]") {
  edm::VecArray<int, 4> array;
  REQUIRE(array.empty());
  REQUIRE(array.size() == 0);
  REQUIRE(array.capacity() == 4);
  REQUIRE((edm::VecArray<int, 4>::capacity() == 4));

  auto iter = array.begin();
  auto end = array.end();
  REQUIRE(iter == end);

  array.push_back_unchecked(1);
  REQUIRE(!array.empty());
  REQUIRE(array.size() == 1);
  REQUIRE(array.front() == 1);
  REQUIRE(array.back() == 1);
  iter = array.begin();
  end = array.end();
  REQUIRE(iter != end);
  REQUIRE(*iter == 1);
  ++iter;
  REQUIRE(iter == end);

  array.emplace_back_unchecked(2);
  REQUIRE(array.size() == 2);
  REQUIRE(array.back() == 2);
  array.push_back(3);
  REQUIRE(array.size() == 3);
  REQUIRE(array.back() == 3);
  array.emplace_back(4);
  REQUIRE(array.size() == 4);
  REQUIRE(array.front() == 1);
  REQUIRE(array.back() == 4);
  REQUIRE((array[0] == 1 && array[1] == 2 && array[2] == 3 && array[3] == 4));

  REQUIRE_THROWS_AS(array.push_back(5), std::length_error);
  REQUIRE_THROWS_AS(array.emplace_back(5), std::length_error);

  auto ptr = array.data();
  REQUIRE((ptr[0] == 1 && ptr[1] == 2 && ptr[2] == 3 && ptr[3] == 4));

  testIterators(array);
  testIterators(const_cast<const decltype(array)&>(array));

  edm::VecArray<int, 4> array2;
  array2.push_back(11);
  array2.push_back(12);

  array.swap(array2);
  REQUIRE(array.size() == 2);
  REQUIRE(array2.size() == 4);
  REQUIRE((array[0] == 11 && array[1] == 12));
  REQUIRE((array2[0] == 1 && array2[1] == 2 && array2[2] == 3 && array2[3] == 4));

  array = array2;
  REQUIRE(array.size() == array2.size());
  REQUIRE(array.size() == 4);
  REQUIRE((array[0] == 1 && array[1] == 2 && array[2] == 3 && array[3] == 4));
  REQUIRE((array2[0] == 1 && array2[1] == 2 && array2[2] == 3 && array2[3] == 4));

  ptr = array.data();
  ptr[1] = 10;
  iter = array.begin() + 1;
  REQUIRE(*iter == 10);

  iter = array.begin() + 2;
  *iter = 50;
  REQUIRE((array[0] == 1 && array[1] == 10 && array[2] == 50 && array[3] == 4));

  REQUIRE(!array.empty());
  array.clear();
  REQUIRE(array.empty());
  REQUIRE(array.size() == 0);
  REQUIRE(array2.size() == 4);

  array.resize(2);
  REQUIRE(array.size() == 2);
  array.pop_back();
  REQUIRE(array.size() == 1);
  REQUIRE_THROWS_AS(array.resize(6), std::length_error);
  REQUIRE(array.size() == 1);
  array.resize(4);
  REQUIRE(array.size() == 4);
  array.resize(1);
  REQUIRE(array.size() == 1);
}
