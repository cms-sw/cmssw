#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "DataFormats/Common/interface/StdArray.h"

TEST_CASE("StdArray", "[StdArray]") {
  SECTION("Empty array") {
    edm::StdArray<int, 0> array;
    REQUIRE(array.empty());
    REQUIRE(array.begin() == array.end());
    REQUIRE(array.rbegin() == array.rend());
    for ([[maybe_unused]] auto element : array) {
      REQUIRE(false);
    }
    REQUIRE(std::data(array) == nullptr);
    REQUIRE(std::size(array) == 0);
  }

  SECTION("Default constructor") {
    edm::StdArray<int, 4> array;
    REQUIRE(not array.empty());
    REQUIRE(array.size() == 4);
    REQUIRE(std::data(array) != nullptr);
    REQUIRE(std::size(array) != 0);
  }

  SECTION("Aggregate constructor") {
    edm::StdArray<int, 4> array{{0, 1, 2, 3}};
    REQUIRE(not array.empty());
    REQUIRE(array.size() == 4);
    REQUIRE(array.front() == 0);
    REQUIRE(array.back() == 3);
    for (int i = 0; i < 4; ++i) {
      REQUIRE(array.at(i) == i);
      REQUIRE(array[i] == i);
    }
    REQUIRE(std::data(array) != nullptr);
    REQUIRE(std::size(array) != 0);
  }

  SECTION("Copy constructor") {
    edm::StdArray<int, 4> array{{0, 1, 2, 3}};
    edm::StdArray<int, 4> other = array;
    for (int i = 0; i < 4; ++i) {
      REQUIRE(other[i] == i);
    }
  }

  SECTION("Move constructor") {
    edm::StdArray<int, 4> array{{0, 1, 2, 3}};
    edm::StdArray<int, 4> other = std::move(array);
    for (int i = 0; i < 4; ++i) {
      REQUIRE(other[i] == i);
    }
  }

  SECTION("Copy assignment") {
    edm::StdArray<int, 4> array{{0, 1, 2, 3}};
    edm::StdArray<int, 4> other;
    other = array;
    for (int i = 0; i < 4; ++i) {
      REQUIRE(other[i] == i);
    }
  }

  SECTION("Move assignment") {
    edm::StdArray<int, 4> array{{0, 1, 2, 3}};
    edm::StdArray<int, 4> other;
    other = std::move(array);
    for (int i = 0; i < 4; ++i) {
      REQUIRE(other[i] == i);
    }
  }

  SECTION("Assignment from std::array") {
    edm::StdArray<int, 4> array;
    std::array<int, 4> other{{0, 1, 2, 3}};
    array = other;
    for (int i = 0; i < 4; ++i) {
      REQUIRE(array[i] == i);
    }
  }

  SECTION("Assignment to std::array") {
    edm::StdArray<int, 4> array{{0, 1, 2, 3}};
    std::array<int, 4> other = array;
    for (int i = 0; i < 4; ++i) {
      REQUIRE(other[i] == i);
    }
  }
}
