#include <catch.hpp>

#include "DataFormats/Portable/interface/PortableObject.h"
#include "DataFormats/Portable/interface/PortableHostObject.h"

namespace {
  struct Test {
    int a;
    float b;
  };

  constexpr auto s_tag = "[PortableObject]";
}  // namespace

// This test is currently mostly about the code compiling
TEST_CASE("Use of PortableObject<T> on host code", s_tag) {
  PortableObject<Test, alpaka::DevCpu> obj(cms::alpakatools::host());
  obj->a = 42;

  SECTION("Tests") { REQUIRE(obj->a == 42); }

  static_assert(std::is_same_v<PortableObject<Test, alpaka::DevCpu>, PortableHostObject<Test>>);
}
