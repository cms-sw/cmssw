#include <memory>
#include <tuple>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

enum class TestEnum : int16_t { s0 = -2, s1 = -1, s2 = 0, s3 = 1, s4 = 2 };

// clang-format off
GENERATE_SOA_LAYOUT(SimpleLayoutTemplate,
  SOA_COLUMN(float, x),
  SOA_COLUMN(float, y),
  SOA_COLUMN(float, z),
  SOA_SCALAR(int, s),
  SOA_COLUMN(float, t),
  SOA_COLUMN(TestEnum, e))
// clang-format on

using SimpleLayout = SimpleLayoutTemplate<>;

namespace {
  template <typename TView>
  concept Immutable = requires(TView view) { requires !requires { view[0] = decltype(view[0]){}; }; };
}  // namespace

// Check access operator of columns
template <typename T>
concept CanAssignX = requires(T view) {
  view[0].x() = 1.0;
  view.x(0) = 1.0;
  view.x()[0] = 1.0;
};

TEST_CASE("SoATemplate") {
  // number of elements
  const std::size_t slSize = 10;
  // size in bytes
  const std::size_t slBufferSize = SimpleLayout::computeDataSize(slSize);
  // memory buffer aligned according to the layout requirements
  std::unique_ptr<std::byte, decltype(std::free) *> slBuffer{
      reinterpret_cast<std::byte *>(aligned_alloc(SimpleLayout::alignment, slBufferSize)), std::free};
  // SoA layout
  SimpleLayout sl{slBuffer.get(), slSize};

  SECTION("Row wide copies, row") {
    SimpleLayout::View slv{sl};
    SimpleLayout::ConstView slcv{sl};
    slv.s() = 42;
    auto slv0 = slv[0];
    slv0.x() = 1;
    slv0.y() = 2;
    slv0.z() = 3;
    slv0.t() = 5;
    slv0.e() = TestEnum::s3;
    // Fill up
    for (SimpleLayout::View::size_type i = 1; i < slv.metadata().size(); ++i) {
      auto slvi = slv[i];
      slvi = slv[i - 1];
      auto slvix = slvi.x();
      slvi.x() += slvi.y();
      slvi.y() += slvi.z();
      slvi.z() += slvi.t();
      slvi.t() += slvix;
    }
    // Verification and const view access
    float x = 1, y = 2, z = 3, t = 5;
    REQUIRE(slcv.s() == 42);
    for (SimpleLayout::View::size_type i = 0; i < slv.metadata().size(); ++i) {
      auto slvi = slv[i];
      auto slcvi = slcv[i];

      // check that SCALAR accessors are not available an SoA element
      STATIC_REQUIRE(![](auto &x) { return requires { x.s(); }; }(slvi));
      STATIC_REQUIRE(![](auto &x) { return requires { x.s(); }; }(slcvi));

      REQUIRE(slvi.x() == x);
      REQUIRE(slvi.y() == y);
      REQUIRE(slvi.z() == z);
      REQUIRE(slvi.t() == t);
      REQUIRE(slcvi.x() == x);
      REQUIRE(slcvi.y() == y);
      REQUIRE(slcvi.z() == z);
      REQUIRE(slcvi.t() == t);
      REQUIRE(slcvi.e() == TestEnum::s3);
      auto tx = x;
      x += y;
      y += z;
      z += t;
      t += tx;
    }
  }

  SECTION("Row initializer, const view access, restrict disabling") {
    // With two views, we are creating (artificially) aliasing and should warn the compiler by turning restrict qualifiers off.
    using View = SimpleLayout::ViewTemplate<cms::soa::RestrictQualify::disabled, cms::soa::RangeChecking::Default>;
    using ConstView =
        SimpleLayout::ConstViewTemplate<cms::soa::RestrictQualify::disabled, cms::soa::RangeChecking::Default>;
    View slv{sl};
    ConstView slcv{sl};
    auto slv0 = slv[0];
    slv0 = {7, 11, 13, 17, TestEnum::s3};
    // Fill up
    for (SimpleLayout::View::size_type i = 1; i < slv.metadata().size(); ++i) {
      auto slvi = slv[i];
      // Copy the const view
      slvi = slcv[i - 1];
      auto slvix = slvi.x();
      slvi.x() += slvi.y();
      slvi.y() += slvi.z();
      slvi.z() += slvi.t();
      slvi.t() += slvix;
    }
    // Verification and const view access
    auto [x, y, z, t] = std::make_tuple(7.0, 11.0, 13.0, 17.0);
    for (SimpleLayout::View::size_type i = 0; i < slv.metadata().size(); ++i) {
      auto slvi = slv[i];
      auto slcvi = slcv[i];
      REQUIRE(slvi.x() == x);
      REQUIRE(slvi.y() == y);
      REQUIRE(slvi.z() == z);
      REQUIRE(slvi.t() == t);
      REQUIRE(slcvi.x() == x);
      REQUIRE(slcvi.y() == y);
      REQUIRE(slcvi.z() == z);
      REQUIRE(slcvi.t() == t);
      auto tx = x;
      x += y;
      y += z;
      z += t;
      t += tx;
    }
  }

  SECTION("Range checking View") {
    // Enable range checking
    using View = SimpleLayout::ViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::enabled>;
    View slv{sl};
    int underflow = -1;
    int overflow = slv.metadata().size();
    // Check for under-and overflow in the row accessor
    REQUIRE_THROWS_AS(slv[underflow], std::out_of_range);
    REQUIRE_THROWS_AS(slv[overflow], std::out_of_range);
    // Check for under-and overflow in the element accessors
    REQUIRE_THROWS_AS(slv.x(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(slv.x(overflow), std::out_of_range);
  }

  SECTION("Range checking ConstView") {
    // Enable range checking
    using ConstView =
        SimpleLayout::ConstViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::enabled>;
    ConstView slcv{sl};
    int underflow = -1;
    int overflow = slcv.metadata().size();
    // Check for under-and overflow in the row accessor
    REQUIRE_THROWS_AS(slcv[underflow], std::out_of_range);
    REQUIRE_THROWS_AS(slcv[overflow], std::out_of_range);
    // Check for under-and overflow in the element accessors
    REQUIRE_THROWS_AS(slcv.x(underflow), std::out_of_range);
    REQUIRE_THROWS_AS(slcv.x(overflow), std::out_of_range);
  }

  SECTION("Range checking View Extended") {
    // Enable range checking
    using View = SimpleLayout::ViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::extended>;
    View slv{sl};
    int underflow = -1;
    int overflow = slv.metadata().size();

    REQUIRE_THROWS_WITH(slv[underflow], Catch::Matchers::ContainsSubstring("at file"));
    REQUIRE_THROWS_WITH(slv[overflow], Catch::Matchers::ContainsSubstring("at file"));

    REQUIRE_THROWS_WITH(slv.x(underflow), Catch::Matchers::ContainsSubstring("at file"));
    REQUIRE_THROWS_WITH(slv.x(overflow), Catch::Matchers::ContainsSubstring("at file"));
  }

  SECTION("Range checking ConstView Extended") {
    // Enable range checking
    using ConstView =
        SimpleLayout::ConstViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::extended>;
    ConstView slcv{sl};
    int underflow = -1;
    int overflow = slcv.metadata().size();

    // Check for under- and overflow in the row accessor
    REQUIRE_THROWS_WITH(slcv[underflow], Catch::Matchers::ContainsSubstring("at file"));
    REQUIRE_THROWS_WITH(slcv[overflow], Catch::Matchers::ContainsSubstring("at file"));

    // Check for under- and overflow in the element accessors
    REQUIRE_THROWS_WITH(slcv.x(underflow), Catch::Matchers::ContainsSubstring("at file"));
    REQUIRE_THROWS_WITH(slcv.x(overflow), Catch::Matchers::ContainsSubstring("at file"));
  }

  SECTION("Check immutability of ConstView") {
    using ConstView =
        SimpleLayout::ConstViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::extended>;

    // check that the ConstView itself is mutable
    STATIC_REQUIRE(std::is_assignable_v<ConstView &, ConstView>);

    // check the returned element from the ConstView is immutable
    using ConstElement = decltype(std::declval<ConstView &>()[0]);
    STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<ConstElement>>);
    static_assert(Immutable<ConstView>);

    // check that we can not assign to the column x
    STATIC_REQUIRE_FALSE(CanAssignX<ConstView>);
  }

  SECTION("Check mutability of View") {
    using View = SimpleLayout::ViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::extended>;
    // check that the View itself is mutable
    STATIC_REQUIRE(std::is_assignable_v<View &, View>);

    // check the returned element from the View is immutable
    using Element = decltype(std::declval<View &>()[0]);
    STATIC_REQUIRE_FALSE(std::is_const_v<std::remove_reference_t<Element>>);

    // check that we can assign to the column x
    STATIC_REQUIRE(CanAssignX<View>);
  }

  SECTION("Check views conversions") {
    using CVE_D =
        SimpleLayout::ConstViewTemplate<cms::soa::RestrictQualify::enabled, cms::soa::RangeChecking::disabled>;
    using CVE_E = SimpleLayout::ConstViewTemplate<cms::soa::RestrictQualify::enabled, cms::soa::RangeChecking::enabled>;
    using CVE_X =
        SimpleLayout::ConstViewTemplate<cms::soa::RestrictQualify::enabled, cms::soa::RangeChecking::extended>;

    using CVD_D =
        SimpleLayout::ConstViewTemplate<cms::soa::RestrictQualify::disabled, cms::soa::RangeChecking::disabled>;
    using CVD_E =
        SimpleLayout::ConstViewTemplate<cms::soa::RestrictQualify::disabled, cms::soa::RangeChecking::enabled>;
    using CVD_X =
        SimpleLayout::ConstViewTemplate<cms::soa::RestrictQualify::disabled, cms::soa::RangeChecking::extended>;

    using VE_D = SimpleLayout::ViewTemplate<cms::soa::RestrictQualify::enabled, cms::soa::RangeChecking::disabled>;
    using VE_E = SimpleLayout::ViewTemplate<cms::soa::RestrictQualify::enabled, cms::soa::RangeChecking::enabled>;
    using VE_X = SimpleLayout::ViewTemplate<cms::soa::RestrictQualify::enabled, cms::soa::RangeChecking::extended>;

    using VD_D = SimpleLayout::ViewTemplate<cms::soa::RestrictQualify::disabled, cms::soa::RangeChecking::disabled>;
    using VD_E = SimpleLayout::ViewTemplate<cms::soa::RestrictQualify::disabled, cms::soa::RangeChecking::enabled>;
    using VD_X = SimpleLayout::ViewTemplate<cms::soa::RestrictQualify::disabled, cms::soa::RangeChecking::extended>;

    // View -> View
    static_assert(std::convertible_to<VE_D, VE_E>);
    static_assert(std::convertible_to<VE_D, VE_X>);
    static_assert(std::convertible_to<VE_D, VD_D>);
    static_assert(std::convertible_to<VE_D, VD_E>);
    static_assert(std::convertible_to<VE_D, VD_X>);

    // View -> ConstView
    static_assert(std::convertible_to<VE_D, CVE_D>);
    static_assert(std::convertible_to<VE_D, CVE_E>);
    static_assert(std::convertible_to<VE_D, CVE_X>);
    static_assert(std::convertible_to<VE_D, CVD_D>);
    static_assert(std::convertible_to<VE_D, CVD_E>);
    static_assert(std::convertible_to<VE_D, CVD_X>);

    static_assert(std::convertible_to<VE_E, CVE_D>);
    static_assert(std::convertible_to<VE_E, CVE_E>);
    static_assert(std::convertible_to<VE_E, CVE_X>);
    static_assert(std::convertible_to<VE_E, CVD_D>);
    static_assert(std::convertible_to<VE_E, CVD_E>);
    static_assert(std::convertible_to<VE_E, CVD_X>);

    static_assert(std::convertible_to<VE_X, CVE_D>);
    static_assert(std::convertible_to<VE_X, CVE_E>);
    static_assert(std::convertible_to<VE_X, CVE_X>);
    static_assert(std::convertible_to<VE_X, CVD_D>);
    static_assert(std::convertible_to<VE_X, CVD_E>);
    static_assert(std::convertible_to<VE_X, CVD_X>);

    static_assert(std::convertible_to<VD_D, CVE_D>);
    static_assert(std::convertible_to<VD_D, CVE_E>);
    static_assert(std::convertible_to<VD_D, CVE_X>);
    static_assert(std::convertible_to<VD_D, CVD_D>);
    static_assert(std::convertible_to<VD_D, CVD_E>);
    static_assert(std::convertible_to<VD_D, CVD_X>);

    static_assert(std::convertible_to<VD_E, CVE_D>);
    static_assert(std::convertible_to<VD_E, CVE_E>);
    static_assert(std::convertible_to<VD_E, CVE_X>);
    static_assert(std::convertible_to<VD_E, CVD_D>);
    static_assert(std::convertible_to<VD_E, CVD_E>);
    static_assert(std::convertible_to<VD_E, CVD_X>);

    static_assert(std::convertible_to<VD_X, CVE_D>);
    static_assert(std::convertible_to<VD_X, CVE_E>);
    static_assert(std::convertible_to<VD_X, CVE_X>);
    static_assert(std::convertible_to<VD_X, CVD_D>);
    static_assert(std::convertible_to<VD_X, CVD_E>);
    static_assert(std::convertible_to<VD_X, CVD_X>);

    // ConstView -> ConstView
    static_assert(std::convertible_to<CVE_D, CVE_E>);
    static_assert(std::convertible_to<CVE_D, CVE_X>);
    static_assert(std::convertible_to<CVE_D, CVD_D>);
    static_assert(std::convertible_to<CVE_D, CVD_E>);
    static_assert(std::convertible_to<CVE_D, CVD_X>);

    // ConstView -> View
    static_assert(!std::convertible_to<CVE_D, VE_D>);
    static_assert(!std::convertible_to<CVE_D, VE_E>);
    static_assert(!std::convertible_to<CVE_D, VE_X>);
    static_assert(!std::convertible_to<CVE_D, VD_D>);
    static_assert(!std::convertible_to<CVE_D, VD_E>);
    static_assert(!std::convertible_to<CVE_D, VD_X>);

    static_assert(!std::convertible_to<CVE_E, VE_D>);
    static_assert(!std::convertible_to<CVE_E, VE_E>);
    static_assert(!std::convertible_to<CVE_E, VE_X>);
    static_assert(!std::convertible_to<CVE_E, VD_D>);
    static_assert(!std::convertible_to<CVE_E, VD_E>);
    static_assert(!std::convertible_to<CVE_E, VD_X>);

    static_assert(!std::convertible_to<CVE_X, VE_D>);
    static_assert(!std::convertible_to<CVE_X, VE_E>);
    static_assert(!std::convertible_to<CVE_X, VE_X>);
    static_assert(!std::convertible_to<CVE_X, VD_D>);
    static_assert(!std::convertible_to<CVE_X, VD_E>);
    static_assert(!std::convertible_to<CVE_X, VD_X>);

    static_assert(!std::convertible_to<CVD_D, VE_D>);
    static_assert(!std::convertible_to<CVD_D, VE_E>);
    static_assert(!std::convertible_to<CVD_D, VE_X>);
    static_assert(!std::convertible_to<CVD_D, VD_D>);
    static_assert(!std::convertible_to<CVD_D, VD_E>);
    static_assert(!std::convertible_to<CVD_D, VD_X>);

    static_assert(!std::convertible_to<CVD_E, VE_D>);
    static_assert(!std::convertible_to<CVD_E, VE_E>);
    static_assert(!std::convertible_to<CVD_E, VE_X>);
    static_assert(!std::convertible_to<CVD_E, VD_D>);
    static_assert(!std::convertible_to<CVD_E, VD_E>);
    static_assert(!std::convertible_to<CVD_E, VD_X>);

    static_assert(!std::convertible_to<CVD_X, VE_D>);
    static_assert(!std::convertible_to<CVD_X, VE_E>);
    static_assert(!std::convertible_to<CVD_X, VE_X>);
    static_assert(!std::convertible_to<CVD_X, VD_D>);
    static_assert(!std::convertible_to<CVD_X, VD_E>);
    static_assert(!std::convertible_to<CVD_X, VD_X>);
  }
}
