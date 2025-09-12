/*
 *  CMSSW
 */
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "DataFormats/Common/interface/MultiSpan.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <iostream>
#include <vector>

using namespace edm;

TEST_CASE("MultiSpan basic indexing", "[MultiSpan]") {
  edm::MultiSpan<int> ms;

  std::vector<int> a = {1, 2, 3};
  std::vector<int> b = {4, 5};

  ms.add(a);
  ms.add(b);

  using ElementType = decltype(ms[0]);
  // Check that the const-correctness of the MultiSpan
  static_assert(!std::is_assignable<ElementType, int>::value,
                "It should not be possible to assign to an element of MultiSpan; See PR #48826");

  SECTION("Size is correct") { REQUIRE(ms.size() == 5); }

  SECTION("Indexing returns correct values") {
    REQUIRE(ms[0] == 1);
    REQUIRE(ms[1] == 2);
    REQUIRE(ms[2] == 3);
    REQUIRE(ms[3] == 4);
    REQUIRE(ms[4] == 5);
  }

  SECTION("Global index from span index and local index") {
    REQUIRE(ms.globalIndex(0, 0) == 0);
    REQUIRE(ms.globalIndex(0, 2) == 2);
    REQUIRE(ms.globalIndex(1, 0) == 3);
    REQUIRE(ms.globalIndex(1, 1) == 4);
  }

  SECTION("Span and local index from global index") {
    auto [span0, local0] = ms.spanAndLocalIndex(0);
    REQUIRE(span0 == 0);
    REQUIRE(local0 == 0);

    auto [span1, local1] = ms.spanAndLocalIndex(4);
    REQUIRE(span1 == 1);
    REQUIRE(local1 == 1);
  }

  SECTION("Iterators work with range-based for") {
    std::vector<int> collected;
    for (auto val : ms) {
      collected.push_back(val);
    }
    REQUIRE(collected == std::vector<int>{1, 2, 3, 4, 5});
  }

  SECTION("Random access iterator supports arithmetic") {
    auto it = ms.begin();
    REQUIRE(*(it + 2) == 3);
    REQUIRE(*(ms.end() - 2) == 4);
    REQUIRE((ms.end() - ms.begin()) == 5);
  }

  SECTION("std::find works") {
    auto it = std::find(ms.begin(), ms.end(), 4);
    REQUIRE(it != ms.end());
    REQUIRE(*it == 4);
    REQUIRE((it - ms.begin()) == 3);
  }

  SECTION("std::distance returns correct result") {
    auto dist = std::distance(ms.begin(), ms.end());
    REQUIRE(dist == 5);
  }

  SECTION("std::copy copies all values") {
    std::vector<int> out(5);
    std::copy(ms.begin(), ms.end(), out.begin());

    REQUIRE(ms[0] == out[0]);
    REQUIRE(ms[1] == out[1]);
    REQUIRE(ms[2] == out[2]);
    REQUIRE(ms[3] == out[3]);
    REQUIRE(ms[4] == out[4]);
  }
}
