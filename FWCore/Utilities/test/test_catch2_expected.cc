#include <memory>
#include <string>
#include <type_traits>
#include <vector>

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

  SECTION("error_or converts each alternative to the error type") {
    // the common type of long long and double is double, which cannot represent 2^53 + 1
    constexpr long long large = (1LL << 53) + 1;
    edm::expected<int, long long> success(1);
    edm::expected<int, long long> failure{edm::unexpected(large)};

    CHECK(success.error_or(2.) == 2);
    CHECK(failure.error_or(2.) == large);
    CHECK(std::move(failure).error_or(2.) == large);
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

namespace {
  edm::expected<std::string, int> make_value() { return "value"; }
  edm::expected<std::string, int> make_error() { return edm::unexpected(42); }
  edm::expected<int, std::string> make_string_error() { return edm::unexpected("string error"); }
}  // namespace

TEST_CASE("edm::unexpected", "[expected]") {
  SECTION("Deduction guide") {
    STATIC_CHECK(std::is_same_v<decltype(edm::unexpected(42)), edm::unexpected<int>>);
    STATIC_CHECK(std::is_same_v<decltype(edm::unexpected("error")), edm::unexpected<const char*>>);
  }

  SECTION("Converting construction") {
    edm::unexpected<std::string> u("error");
    CHECK(u.error() == "error");
  }

  SECTION("rvalue error access") {
    edm::unexpected<std::string> u("error");
    std::string err = std::move(u).error();
    CHECK(err == "error");

    const edm::unexpected<std::string> cu("const error");
    STATIC_CHECK(std::is_same_v<decltype(std::move(cu).error()), const std::string&&>);
    CHECK(std::move(cu).error() == "const error");
  }
}

TEST_CASE("edm::expected converting construction", "[expected]") {
  SECTION("Implicit conversion from a value") {
    auto e = make_value();
    REQUIRE(e.has_value());
    CHECK(*e == "value");
  }

  SECTION("Implicit conversion from an unexpected") {
    auto e = make_error();
    REQUIRE_FALSE(e.has_value());
    CHECK(e.error() == 42);
  }

  SECTION("Conversion from an unexpected with a different error type") {
    auto e = make_string_error();
    REQUIRE_FALSE(e.has_value());
    CHECK(e.error() == "string error");

    const edm::unexpected<const char*> u("const error");
    edm::expected<int, std::string> f(u);
    REQUIRE_FALSE(f.has_value());
    CHECK(f.error() == "const error");
  }

  SECTION("Explicit conversions") {
    // std::vector<int> is constructible but not convertible from int
    using Expected = edm::expected<std::vector<int>, std::vector<int>>;
    STATIC_CHECK(std::is_constructible_v<Expected, int>);
    STATIC_CHECK_FALSE(std::is_convertible_v<int, Expected>);
    STATIC_CHECK(std::is_constructible_v<Expected, edm::unexpected<int>>);
    STATIC_CHECK_FALSE(std::is_convertible_v<edm::unexpected<int>, Expected>);

    Expected value(3);
    REQUIRE(value.has_value());
    CHECK(value->size() == 3);

    Expected error(edm::unexpected(2));
    REQUIRE_FALSE(error.has_value());
    CHECK(error.error().size() == 2);
  }

  SECTION("Value and error of the same type") {
    edm::expected<int, int> value(1);
    edm::expected<int, int> error(edm::unexpected(2));
    CHECK(value.has_value());
    CHECK(*value == 1);
    CHECK_FALSE(error.has_value());
    CHECK(error.error() == 2);
  }

  SECTION("Constant evaluation") {
    STATIC_CHECK(edm::expected<int, int>(42).value() == 42);
    STATIC_CHECK(edm::expected<int, int>(edm::unexpected(3)).error() == 3);
  }
}

namespace {
  struct ThrowingMove {
    ThrowingMove() = default;
    ThrowingMove(ThrowingMove const&) = default;
    ThrowingMove(ThrowingMove&&) noexcept(false) {}
    ThrowingMove& operator=(ThrowingMove const&) = default;
    ThrowingMove& operator=(ThrowingMove&&) = default;
  };
}  // namespace

TEST_CASE("edm::expected copy, move and assignment", "[expected]") {
  SECTION("Trivial types stay trivial") {
    // std::expected guarantees trivial copy and move construction and destruction, but not trivial assignment
    STATIC_CHECK(std::is_trivially_copy_constructible_v<edm::expected<int, int>>);
    STATIC_CHECK(std::is_trivially_move_constructible_v<edm::expected<int, int>>);
    STATIC_CHECK(std::is_trivially_destructible_v<edm::expected<int, int>>);
  }

  SECTION("Assignment between value and error") {
    edm::expected<std::string, std::string> e("value");
    e = edm::unexpected<std::string>("error");
    REQUIRE_FALSE(e.has_value());
    CHECK(e.error() == "error");

    edm::expected<std::string, std::string> other("other value");
    e = other;
    REQUIRE(e.has_value());
    CHECK(*e == "other value");

    e = edm::expected<std::string, std::string>(edm::unexpected<std::string>("moved error"));
    REQUIRE_FALSE(e.has_value());
    CHECK(e.error() == "moved error");
  }

  SECTION("Move-only types") {
    edm::expected<std::unique_ptr<int>, int> e(std::make_unique<int>(1));
    edm::expected<std::unique_ptr<int>, int> f(std::move(e));
    REQUIRE(f.has_value());
    CHECK(**f == 1);
    e = std::move(f);
    REQUIRE(e.has_value());
    CHECK(**e == 1);
    STATIC_CHECK_FALSE(std::is_copy_constructible_v<edm::expected<std::unique_ptr<int>, int>>);
  }

  SECTION("Types with a throwing move constructor are not assignable") {
    // Stricter than std::expected: this guarantees that the std::variant storage never becomes valueless.
    // These are runtime checks, because they fail with std::expected.
    using Expected = edm::expected<ThrowingMove, int>;
    STATIC_CHECK(std::is_copy_constructible_v<Expected>);
    STATIC_CHECK(std::is_move_constructible_v<Expected>);
    CHECK_FALSE(std::is_copy_assignable_v<Expected>);
    CHECK_FALSE(std::is_move_assignable_v<Expected>);
  }
}
