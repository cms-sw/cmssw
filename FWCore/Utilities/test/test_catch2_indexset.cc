#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/IndexSet.h"

TEST_CASE("IndexSet basic operations", "[IndexSet]") {
  edm::IndexSet set;
  REQUIRE(set.empty());
  REQUIRE(set.size() == 0);
  REQUIRE(!set.has(0));

  set.reserve(10);

  REQUIRE(set.empty());
  REQUIRE(set.size() == 0);
  REQUIRE(!set.has(0));

  set.insert(0);
  REQUIRE(!set.empty());
  REQUIRE(set.size() == 1);
  REQUIRE(set.has(0));
  REQUIRE(!set.has(1));

  set.insert(2);
  REQUIRE(set.size() == 2);
  REQUIRE(set.has(0));
  REQUIRE(!set.has(1));
  REQUIRE(set.has(2));
  REQUIRE(!set.has(3));

  set.insert(20);
  REQUIRE(set.size() == 3);
  REQUIRE(set.has(0));
  REQUIRE(!set.has(1));
  REQUIRE(set.has(2));
  REQUIRE(!set.has(3));
  REQUIRE(!set.has(19));
  REQUIRE(set.has(20));
  REQUIRE(!set.has(21));

  set.insert(2);
  REQUIRE(set.size() == 3);
  REQUIRE(set.has(2));

  set.clear();
  REQUIRE(set.empty());
  REQUIRE(set.size() == 0);
  REQUIRE(!set.has(0));
  REQUIRE(!set.has(1));
  REQUIRE(!set.has(2));
  REQUIRE(!set.has(3));
}
