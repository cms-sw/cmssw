#include <catch2/catch_all.hpp>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace {
  GENERATE_SOA_LAYOUT(TestLayout, SOA_COLUMN(double, x), SOA_COLUMN(int32_t, id))

  using TestSoA = TestLayout<>;

  constexpr auto s_tag = "[PortableCollection]";
}  // namespace

// This test is currently mostly about the code compiling
TEST_CASE("Use of PortableCollection<T, TDev> on host code", s_tag) {
  auto const size = 10;
  PortableCollection<alpaka::DevCpu, TestSoA> coll(cms::alpakatools::host(), size);

  SECTION("Tests") { REQUIRE(coll->metadata().size() == size); }

  static_assert(std::is_same_v<PortableCollection<alpaka::DevCpu, TestSoA>, PortableHostCollection<TestSoA>>);
}
