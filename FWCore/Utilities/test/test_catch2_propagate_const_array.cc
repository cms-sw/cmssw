/*----------------------------------------------------------------------

Test program for edm::propagate_const_array class.

 ----------------------------------------------------------------------*/

#include <catch2/catch_all.hpp>
#include "FWCore/Utilities/interface/get_underlying_safe.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"

namespace {
  class ConstChecker {
  public:
    int value() { return 0; }

    int const value() const { return 1; }
  };

  // used to check that edm::propagate_const_array<T[]> works with incomplete types
  class Incomplete;

}  // namespace

TEST_CASE("propagate_const_array basic operations", "[propagate_const_array]") {
  SECTION("test propagate_const_array<T[]>") {
    ConstChecker checker[10];
    edm::propagate_const_array<ConstChecker[]> pChecker(checker);

    REQUIRE(0 == pChecker[1].value());
    REQUIRE(0 == pChecker.get()[1].value());
    REQUIRE(0 == get_underlying_safe(pChecker)[1].value());

    const edm::propagate_const_array<ConstChecker[]> pConstChecker(checker);

    REQUIRE(1 == pConstChecker[1].value());
    REQUIRE(1 == pConstChecker.get()[1].value());
    REQUIRE(1 == get_underlying_safe(pConstChecker)[1].value());
  }

  // test for edm::propagate_const_array<shared_ptr<T>>
  SECTION("test for edm::propagate_const_array<shared_ptr<T>>") {
    // std::make_shared<T[]> requires C++20
    auto checker = std::shared_ptr<ConstChecker[]>(new ConstChecker[10]);
    edm::propagate_const_array<std::shared_ptr<ConstChecker[]>> pChecker(checker);

    REQUIRE(0 == pChecker[1].value());
    REQUIRE(0 == pChecker.get()[1].value());
    REQUIRE(0 == get_underlying_safe(pChecker)[1].value());

    const edm::propagate_const_array<std::shared_ptr<ConstChecker[]>> pConstChecker(checker);

    REQUIRE(1 == pConstChecker[1].value());
    REQUIRE(1 == pConstChecker.get()[1].value());
    REQUIRE(1 == get_underlying_safe(pConstChecker)[1].value());
  }

  SECTION("test for edm::propagate_const_array<unique_ptr<T>>") {
    auto checker = std::make_unique<ConstChecker[]>(10);
    edm::propagate_const_array<std::unique_ptr<ConstChecker[]>> pChecker(std::move(checker));

    REQUIRE(0 == pChecker[1].value());
    REQUIRE(0 == pChecker.get()[1].value());
    REQUIRE(0 == get_underlying_safe(pChecker)[1].value());

    const edm::propagate_const_array<std::unique_ptr<ConstChecker[]>> pConstChecker(
        std::make_unique<ConstChecker[]>(10));

    REQUIRE(1 == pConstChecker[1].value());
    REQUIRE(1 == pConstChecker.get()[1].value());
  }

  SECTION("test for edm::propagate_const_array<T[]> with incomplete types") {
    edm::propagate_const_array<Incomplete[]> pIncomplete(nullptr);

    REQUIRE(nullptr == get_underlying(pIncomplete));
  }
}
