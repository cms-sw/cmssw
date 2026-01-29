#include <catch2/catch_all.hpp>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/Portable/interface/PortableCollectionCommon.h"

using namespace portablecollection;

namespace {
  GENERATE_SOA_LAYOUT(TestLayout, SOA_COLUMN(double, x), SOA_COLUMN(int32_t, id))

  using TestSoA = TestLayout<>;

  GENERATE_SOA_BLOCKS(Blocks4Layout,
                      SOA_BLOCK(first, TestLayout),
                      SOA_BLOCK(second, TestLayout),
                      SOA_BLOCK(third, TestLayout),
                      SOA_BLOCK(fourth, TestLayout))

  using Blocks4 = Blocks4Layout<>;

  using TestCollection1 = PortableHostCollection<TestSoA>;
  using TestCollection2 = PortableHostCollection<Blocks4>;
}  // namespace

// Check that invalid inputs for the elements parameter of the PortableHostCollection constructor
// cause narrow_cast to throw a std::runtime_error.
TEST_CASE("checked_int_cast enforces non-negative narrowing to int", "[checked_int_cast]") {
  SECTION("signed integer sources") {
    SECTION("valid signed values") {
      REQUIRE_NOTHROW([&] { TestCollection1 check(0); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(1); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<int8_t>(1)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<int16_t>(2)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<int32_t>(3)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<int64_t>(4)); }());

      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<int_fast8_t>(5)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<int_fast16_t>(6)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<int_fast32_t>(7)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<int_fast64_t>(8)); }());

      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<int_least8_t>(9)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<int_least16_t>(10)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<int_least32_t>(11)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<int_least64_t>(12)); }());

      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<signed char>(13)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<short>(14)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<short int>(15)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<signed short>(16)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<signed short int>(17)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<int>(18)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<signed>(19)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<long>(20)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<long int>(21)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<signed long>(22)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<signed long int>(23)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<long long>(24)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<long long int>(25)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<signed long long>(26)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<signed long long int>(27)); }());
      // To not allocate too much memory we just check the cast not the actual object creation
      REQUIRE_NOTHROW(size_cast(std::numeric_limits<int>::max()));
    }

    SECTION("valid unsigned values") {
      REQUIRE_NOTHROW([&] { TestCollection1 check(0); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(1); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<uint8_t>(1)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<uint16_t>(2)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<uint32_t>(3)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<uint64_t>(4)); }());

      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<uint_fast8_t>(5)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<uint_fast16_t>(6)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<uint_fast32_t>(7)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<uint_fast64_t>(8)); }());

      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<uint_least8_t>(9)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<uint_least16_t>(10)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<uint_least32_t>(11)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<uint_least64_t>(12)); }());

      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<unsigned char>(13)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<unsigned short>(14)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<unsigned short int>(15)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<unsigned>(16)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<unsigned int>(17)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<unsigned long>(18)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<unsigned long int>(19)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<unsigned long long>(20)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<unsigned long long int>(21)); }());
    }

    SECTION("invalid negative values") {
      REQUIRE_THROWS_AS([&] { TestCollection1 check(-1); }(), std::runtime_error);
      REQUIRE_THROWS_AS([&] { TestCollection1 check(-42); }(), std::runtime_error);
      REQUIRE_THROWS_AS([&] { TestCollection1 check(std::numeric_limits<int>::min()); }(), std::runtime_error);
      REQUIRE_THROWS_AS([&] { TestCollection1 check(int8_t{-1}); }(), std::runtime_error);
      REQUIRE_THROWS_AS([&] { TestCollection1 check(int16_t{-1}); }(), std::runtime_error);
      REQUIRE_THROWS_AS([&] { TestCollection1 check(int64_t{-1}); }(), std::runtime_error);
    }

    SECTION("signed values exceeding int max are rejected") {
      REQUIRE_THROWS_AS([&] { TestCollection1 check(static_cast<long long>(std::numeric_limits<int>::max()) + 1); }(),
                        std::runtime_error);

      REQUIRE_THROWS_AS([&] { TestCollection1 check(std::numeric_limits<int64_t>::max()); }(), std::runtime_error);
    }
  }

  SECTION("character types") {
    SECTION("signed char") {
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<signed char>(0)); }());
      REQUIRE_THROWS_AS([&] { TestCollection1 check(static_cast<signed char>(-1)); }(), std::runtime_error);
    }

    SECTION("unsigned char") {
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<unsigned char>(0)); }());
      REQUIRE_NOTHROW([&] { TestCollection1 check(static_cast<unsigned char>(255)); }());
    }
  }

  SECTION("BlocksCollection") {
    REQUIRE_NOTHROW([&] {
      TestCollection2 check(
          static_cast<int8_t>(1), static_cast<int16_t>(2), static_cast<int32_t>(3), static_cast<int64_t>(42));
    }());

    REQUIRE_NOTHROW([&] {
      TestCollection2 check(
          static_cast<uint8_t>(1), static_cast<uint16_t>(2), static_cast<uint32_t>(3), static_cast<uint64_t>(42));
    }());

    REQUIRE_THROWS_AS(
        [&] {
          TestCollection2 check(
              static_cast<uint8_t>(1), static_cast<uint16_t>(2), static_cast<uint32_t>(3), static_cast<int64_t>(-1));
        }(),
        std::runtime_error);
  }
}
