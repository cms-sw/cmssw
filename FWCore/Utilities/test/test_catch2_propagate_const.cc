/*----------------------------------------------------------------------

Test program for edm::propagate_const class.

 ----------------------------------------------------------------------*/

#include <catch2/catch_all.hpp>
#include <memory>
#include "FWCore/Utilities/interface/get_underlying_safe.h"
#include "FWCore/Utilities/interface/propagate_const.h"

namespace {
  class ConstChecker {
  public:
    int value() { return 0; }

    int const value() const { return 1; }
  };

  // used to check that edm::propagate_const<T*> works with incomplete types
  class Incomplete;

}  // namespace
TEST_CASE("propagate_const basic operations", "[propagate_const]") {
  SECTION("test for edm::propagate_const<T*>") {
    ConstChecker checker;
    edm::propagate_const<ConstChecker*> pChecker(&checker);

    REQUIRE(0 == pChecker.get()->value());
    REQUIRE(pChecker.get()->value() == pChecker->value());
    REQUIRE(pChecker.get()->value() == (*pChecker).value());
    REQUIRE(0 == get_underlying_safe(pChecker)->value());

    const edm::propagate_const<ConstChecker*> pConstChecker(&checker);

    REQUIRE(1 == pConstChecker.get()->value());
    REQUIRE(pConstChecker.get()->value() == pConstChecker->value());
    REQUIRE(pConstChecker.get()->value() == (*pConstChecker).value());
    REQUIRE(1 == get_underlying_safe(pConstChecker)->value());
  }

  SECTION("test for edm::propagate_const<shared_ptr<T>>") {
    auto checker = std::make_shared<ConstChecker>();
    edm::propagate_const<std::shared_ptr<ConstChecker>> pChecker(checker);

    REQUIRE(0 == pChecker.get()->value());
    REQUIRE(pChecker.get()->value() == pChecker->value());
    REQUIRE(pChecker.get()->value() == (*pChecker).value());
    REQUIRE(0 == get_underlying_safe(pChecker)->value());

    const edm::propagate_const<std::shared_ptr<ConstChecker>> pConstChecker(checker);

    REQUIRE(1 == pConstChecker.get()->value());
    REQUIRE(pConstChecker.get()->value() == pConstChecker->value());
    REQUIRE(pConstChecker.get()->value() == (*pConstChecker).value());
    REQUIRE(1 == get_underlying_safe(pConstChecker)->value());
  }

  // test for edm::propagate_const<unique_ptr<T>>
  SECTION("test for edm::propagate_const<unique_ptr<T>>") {
    auto checker = std::make_unique<ConstChecker>();
    edm::propagate_const<std::unique_ptr<ConstChecker>> pChecker(std::move(checker));

    REQUIRE(0 == pChecker.get()->value());
    REQUIRE(pChecker.get()->value() == pChecker->value());
    REQUIRE(pChecker.get()->value() == (*pChecker).value());
    REQUIRE(0 == get_underlying_safe(pChecker)->value());

    const edm::propagate_const<std::unique_ptr<ConstChecker>> pConstChecker(std::make_unique<ConstChecker>());

    REQUIRE(1 == pConstChecker.get()->value());
    REQUIRE(pConstChecker.get()->value() == pConstChecker->value());
    REQUIRE(pConstChecker.get()->value() == (*pConstChecker).value());
  }

  SECTION("test for edm::propagate_const<T*> with incomplete types") {
    Incomplete* ptr = nullptr;
    edm::propagate_const<Incomplete*> pIncomplete(ptr);

    REQUIRE(nullptr == pIncomplete.get());
  }
}
