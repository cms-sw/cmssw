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
  static_assert(std::is_same_v<PortableObject<Test, alpaka::DevCpu>, PortableHostObject<Test>>);

  SECTION("Initialize by setting members") {
    SECTION("With device") {
      PortableObject<Test, alpaka::DevCpu> obj(cms::alpakatools::host());
      obj->a = 42;

      REQUIRE(obj->a == 42);
    }

    SECTION("With queue") {
      alpaka::QueueCpuBlocking queue(cms::alpakatools::host());

      PortableObject<Test, alpaka::DevCpu> obj(queue);
      obj->a = 42;

      REQUIRE(obj->a == 42);
    }
  }

  SECTION("Initialize via constructor") {
    SECTION("With device") {
      PortableObject<Test, alpaka::DevCpu> obj(cms::alpakatools::host(), Test{42, 3.14f});

      REQUIRE(obj->a == 42);
      REQUIRE(obj->b == 3.14f);
    }

    SECTION("With queue") {
      alpaka::QueueCpuBlocking queue(cms::alpakatools::host());
      PortableObject<Test, alpaka::DevCpu> obj(queue, Test{42, 3.14f});

      REQUIRE(obj->a == 42);
      REQUIRE(obj->b == 3.14f);
    }
  }
}
