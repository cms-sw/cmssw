#include <string>
#include <memory>

#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/expected.h"

TEST_CASE("edm::expected construction and state", "[expected]") {
  SECTION("Construct with value") {
    edm::expected<int, std::string> e(42);
    REQUIRE(e.has_value());
    REQUIRE(static_cast<bool>(e) == true);
    CHECK(*e == 42);
  }

  SECTION("Construct with unexpected") {
    edm::expected<int, std::string> e(edm::unexpected<std::string>("error"));
    REQUIRE_FALSE(e.has_value());
    REQUIRE(static_cast<bool>(e) == false);
    CHECK(e.error() == "error");
  }

  SECTION("Move construction of value") {
    auto ptr = std::make_unique<int>(100);
    edm::expected<std::unique_ptr<int>, int> e(std::move(ptr));
    CHECK(e.has_value());
    CHECK(**e == 100);
    CHECK(ptr == nullptr);  // Ensure it was actually moved
  }
}

TEST_CASE("edm::expected value access", "[expected]") {
  edm::expected<std::string, int> e("hello");

  SECTION("Pointer and Reference operators") {
    CHECK(e->size() == 5);
    CHECK(*e == "hello");
    *e = "world";
    CHECK(*e == "world");
  }

  SECTION("Method value() success") { CHECK(e.value() == "hello"); }

  SECTION("Method value() throws on error") {
    edm::expected<int, std::string> err_exp(edm::unexpected<std::string>("fail"));
    REQUIRE_THROWS_AS(err_exp.value(), std::logic_error);
  }

  SECTION("rvalue value() move semantics") {
    edm::expected<std::string, int> move_me("move");
    std::string target = std::move(move_me).value();
    CHECK(target == "move");
    // Note: move_me is now in a valid but unspecified state
  }
}

TEST_CASE("edm::expected error access", "[expected]") {
  edm::expected<int, std::string> e(edm::unexpected<std::string>("failure"));

  SECTION("lvalue error access") {
    CHECK(e.error() == "failure");
    e.error() = "new failure";
    CHECK(e.error() == "new failure");
  }

  SECTION("rvalue error access") {
    std::string err = std::move(e).error();
    CHECK(err == "failure");
    // Note: e.error() is now in a valid but unspecified state
  }
}

TEST_CASE("edm::expected value_or / error_or", "[expected]") {
  SECTION("value_or logic") {
    edm::expected<int, int> success(1);
    edm::expected<int, int> failure(edm::unexpected<int>(0));

    CHECK(success.value_or(100) == 1);
    CHECK(failure.value_or(100) == 100);
  }

  SECTION("error_or logic") {
    edm::expected<int, std::string> success(1);
    edm::expected<int, std::string> failure(edm::unexpected<std::string>("real error"));

    CHECK(success.error_or("default") == "default");
    CHECK(failure.error_or("default") == "real error");
  }
}

TEST_CASE("edm::expected C++20 Constraints", "[expected][concepts]") {
  // This section is more about ensuring the code compiles/doesn't compile
  // as expected. You can't easily test "failure to compile" inside a test runner,
  // but we verify valid types work.

  struct Success {};
  struct Error {};

  STATIC_CHECK(requires { typename edm::expected<Success, Error>; });

  // The following would fail to compile due to the 'requires' clause:
  // typename edm::expected<edm::unexpected<int>, int> invalid;
}
