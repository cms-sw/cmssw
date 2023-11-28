#include <memory>
#include <tuple>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

// clang-format off
GENERATE_SOA_LAYOUT(SimpleLayoutTemplate,
  SOA_COLUMN(float, x),
  SOA_COLUMN(float, y),
  SOA_COLUMN(float, z),
  SOA_COLUMN(float, t))
// clang-format on

using SimpleLayout = SimpleLayoutTemplate<>;

TEST_CASE("SoATemplate") {
  const std::size_t slSize = 10;
  const std::size_t slBufferSize = SimpleLayout::computeDataSize(slSize);
  std::unique_ptr<std::byte, decltype(std::free) *> slBuffer{
      reinterpret_cast<std::byte *>(aligned_alloc(SimpleLayout::alignment, slBufferSize)), std::free};
  SimpleLayout sl{slBuffer.get(), slSize};
  SECTION("Row wide copies, row") {
    SimpleLayout::View slv{sl};
    SimpleLayout::ConstView slcv{sl};
    auto slv0 = slv[0];
    slv0.x() = 1;
    slv0.y() = 2;
    slv0.z() = 3;
    slv0.t() = 5;
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

  SECTION("Row initializer, const view access, restrict disabling") {
    // With two views, we are creating (artificially) aliasing and should warn the compiler by turning restrict qualifiers off.
    using View = SimpleLayout::ViewTemplate<cms::soa::RestrictQualify::disabled, cms::soa::RangeChecking::Default>;
    using ConstView =
        SimpleLayout::ConstViewTemplate<cms::soa::RestrictQualify::disabled, cms::soa::RangeChecking::Default>;
    View slv{sl};
    ConstView slcv{sl};
    auto slv0 = slv[0];
    slv0 = {7, 11, 13, 17};
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
}