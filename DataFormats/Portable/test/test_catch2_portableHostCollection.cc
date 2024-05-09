#include <catch.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace {
  GENERATE_SOA_LAYOUT(TestLayout, SOA_COLUMN(double, x), SOA_COLUMN(int32_t, id), SOA_SCALAR(uint32_t, num))

  using TestSoA = TestLayout<>;

  constexpr auto s_tag = "[PortableHostCollection]";
}  // namespace

TEST_CASE("PortableHostCollection<T>", s_tag) {
  SECTION("Default constructor") {
    PortableHostCollection<TestSoA> coll;
    REQUIRE(coll.size() == 0);
    REQUIRE(not coll.isValid());

    // Following lines would be undefined behavior, and could lead to crashes
    //coll->num() = 42;
    //REQUIRE(coll->num() == 42);
  }

  SECTION("Zero size") {
    int constexpr size = 0;
    PortableHostCollection<TestSoA> coll(size, cms::alpakatools::host());
    REQUIRE(coll.size() == size);
    REQUIRE(coll.isValid());

    coll->num() = 42;
    REQUIRE(coll->num() == 42);
  }

  SECTION("Non-zero size") {
    int constexpr size = 10;
    PortableHostCollection<TestSoA> coll(size, cms::alpakatools::host());
    REQUIRE(coll.size() == size);
    REQUIRE(coll.isValid());

    coll->num() = 42;
    for (int i = 0; i < size; ++i) {
      coll->id()[i] = i * 2;
    }

    REQUIRE(coll->num() == 42);
    for (int i = 0; i < size; ++i) {
      REQUIRE(coll->id()[i] == i * 2);
    }

    SECTION("Move constructor") {
      PortableHostCollection<TestSoA> coll2(std::move(coll));
      REQUIRE(coll2.size() == size);
      REQUIRE(coll2.isValid());

      REQUIRE(coll.size() == 0);
      REQUIRE(not coll.isValid());
    }

    SECTION("Move assignment") {
      PortableHostCollection<TestSoA> coll2(2*size, cms::alpakatools::host());
      REQUIRE(coll2.size() == 2*size);
      REQUIRE(coll2.isValid());

      coll2 = std::move(coll);
      REQUIRE(coll2.size() == size);
      REQUIRE(coll2.isValid());

      REQUIRE(coll.size() == 0);
      REQUIRE(not coll.isValid());

      SECTION("Self assignment") {
        coll2 = std::move(coll2);
        REQUIRE(coll2.size() == size);
        REQUIRE(coll2.isValid());

        REQUIRE(coll2->num() == 42);
        for (int i = 0; i < size; ++i) {
          REQUIRE(coll2->id()[i] == i * 2);
        }
      }
    }
  }
}
