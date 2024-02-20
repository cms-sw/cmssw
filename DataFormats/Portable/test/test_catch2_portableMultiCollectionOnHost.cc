#include <catch.hpp>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace {
  GENERATE_SOA_LAYOUT(TestLayout1, SOA_COLUMN(double, x), SOA_COLUMN(int32_t, id))
  GENERATE_SOA_LAYOUT(TestLayout2, SOA_COLUMN(float, y), SOA_COLUMN(int32_t, z))

  using TestSoA1 = TestLayout1<>;
  using TestSoA2 = TestLayout2<>;

  constexpr auto s_tag = "[PortableMultiCollection]";
}  // namespace

// This test is currently mostly about the code compiling
TEST_CASE("Use of PortableMultiCollection<T, TDev> on host code", s_tag) {
  std::array<int, 2> const sizes{{10, 5}};

  PortableMultiCollection<alpaka::DevCpu, TestSoA1, TestSoA2> coll(sizes, cms::alpakatools::host());

  SECTION("Tests") { REQUIRE(coll.sizes() == sizes); }

  static_assert(std::is_same_v<PortableMultiCollection<alpaka::DevCpu, TestSoA1, TestSoA2>,
                               PortableHostMultiCollection<TestSoA1, TestSoA2>>);
}
