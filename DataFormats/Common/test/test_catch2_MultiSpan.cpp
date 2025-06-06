#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>
#include <catch2/catch_all.hpp>

#include "DataFormats/Common/interface/MultiSpan.h"
#include "DataFormats/Common/interface/RefProd.h"

TEST_CASE("MultiSpan basic indexing", "[MultiSpan]") {
  edm::MultiSpan<int> emptyMultiSpan;
  edm::MultiSpan<int> ms;

  edm::MultiSpan<int> ms1;  // MultiSpan with empty span as first span
  edm::MultiSpan<int> ms2;  // MultiSpan with several empty spans
  edm::MultiSpan<int> ms3;  // MultiSpan with empty span as last span

  std::vector<int> a = {1, 2, 3};
  std::vector<int> b = {4, 5};
  std::vector<int> c;

  ms.add(a);
  ms.add(b);

  ms1.add(c);
  ms1.add(b);
  ms1.add(a);

  ms2.add(b);
  ms2.add(c);
  ms2.add(c);
  ms2.add(c);
  ms2.add(b);
  ms2.add(a);
  ms2.add(c);

  ms3.add(a);
  ms3.add(c);

  std::vector<edm::RefProd<std::vector<int>>> refProducts;
  refProducts.push_back(edm::RefProd<std::vector<int>>(&b));
  refProducts.push_back(edm::RefProd<std::vector<int>>(&c));
  refProducts.push_back(edm::RefProd<std::vector<int>>(&c));
  refProducts.push_back(edm::RefProd<std::vector<int>>(&c));
  refProducts.push_back(edm::RefProd<std::vector<int>>(&b));
  refProducts.push_back(edm::RefProd<std::vector<int>>(&a));
  refProducts.push_back(edm::RefProd<std::vector<int>>(&c));
  edm::MultiSpan<int> ms4(refProducts);  // MultiSpan from std::vector of RefProds<std::vector>

  using ElementType = decltype(ms[0]);
  // Check that the const-correctness of the MultiSpan
  static_assert(!std::is_assignable<ElementType, int>::value,
                "It should not be possible to assign to an element of MultiSpan; See PR #48826");

  SECTION("Empty MultiSpan") {
    REQUIRE(emptyMultiSpan.size() == 0);
    REQUIRE(emptyMultiSpan.begin() == emptyMultiSpan.end());
    REQUIRE_THROWS_AS(emptyMultiSpan[0], std::out_of_range);
    REQUIRE_THROWS_AS(emptyMultiSpan.globalIndex(0, 0), std::out_of_range);
  }

  SECTION("Size is correct") { REQUIRE(ms.size() == 5); }

  SECTION("Range check") {
    REQUIRE_THROWS_AS(ms[5], std::out_of_range);
    REQUIRE_THROWS_AS(ms.globalIndex(2, 0), std::out_of_range);
    REQUIRE_THROWS_AS(ms.globalIndex(1, 2), std::out_of_range);
    REQUIRE_THROWS_AS(ms.spanAndLocalIndex(5), std::out_of_range);
  }

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

  SECTION("Check MultiSpan with empty span as first span") {
    REQUIRE(ms1.size() == 5);

    REQUIRE(ms1[0] == b[0]);
    REQUIRE(ms1[1] == b[1]);
    REQUIRE(ms1[2] == a[0]);
    REQUIRE(ms1[3] == a[1]);
    REQUIRE(ms1[4] == a[2]);

    REQUIRE(ms1.globalIndex(0, 0) == 0);
    REQUIRE(ms1.globalIndex(1, 1) == 3);

    std::vector<int> collected;
    for (auto val : ms1) {
      collected.push_back(val);
    }
    REQUIRE(collected == std::vector<int>{b[0], b[1], a[0], a[1], a[2]});
  }

  SECTION("Check MultiSpan with serveral empty spans") {
    REQUIRE(ms2.size() == 7);

    REQUIRE(ms2[0] == b[0]);
    REQUIRE(ms2[1] == b[1]);
    REQUIRE(ms2[2] == b[0]);
    REQUIRE(ms2[3] == b[1]);
    REQUIRE(ms2[4] == a[0]);
    REQUIRE(ms2[5] == a[1]);
    REQUIRE(ms2[6] == a[2]);

    REQUIRE(ms2.globalIndex(0, 0) == 0);
    REQUIRE(ms2.globalIndex(1, 1) == 3);
    REQUIRE(ms2.globalIndex(2, 1) == 5);

    std::vector<int> collected;
    for (auto val : ms2) {
      collected.push_back(val);
    }
    REQUIRE(collected == std::vector<int>{b[0], b[1], b[0], b[1], a[0], a[1], a[2]});
  }

  SECTION("Check MultiSpan with serveral empty spans") {
    REQUIRE(ms3.size() == 3);

    REQUIRE(ms3[0] == a[0]);
    REQUIRE(ms3[1] == a[1]);
    REQUIRE(ms3[2] == a[2]);

    std::vector<int> collected;
    for (auto val : ms3) {
      collected.push_back(val);
    }
    REQUIRE(collected == std::vector<int>{a[0], a[1], a[2]});
  }

  SECTION("Check MultiSpan constructed from std::vector of RefProds<std::vector>") {
    REQUIRE(ms4.size() == 7);

    REQUIRE(ms4[0] == b[0]);
    REQUIRE(ms4[1] == b[1]);
    REQUIRE(ms4[2] == b[0]);
    REQUIRE(ms4[3] == b[1]);
    REQUIRE(ms4[4] == a[0]);
    REQUIRE(ms4[5] == a[1]);
    REQUIRE(ms4[6] == a[2]);

    REQUIRE(ms4.globalIndex(0, 0) == 0);
    REQUIRE(ms4.globalIndex(1, 1) == 3);
    REQUIRE(ms4.globalIndex(2, 1) == 5);

    std::vector<int> collected;
    for (auto val : ms4) {
      collected.push_back(val);
    }
    REQUIRE(collected == std::vector<int>{b[0], b[1], b[0], b[1], a[0], a[1], a[2]});
  }
}
