#include "catch2/catch_all.hpp"

#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "SimpleEDProductGetter.h"

namespace {
  // the product getter that RefCores read on this thread would use
  edm::EDProductGetter const* currentGetter() {
    edm::EDProductGetter const* getter = nullptr;
    edm::EDProductGetter::assignEDProductGetter(getter);
    return getter;
  }
}  // namespace

TEST_CASE("RefCoreStreamerGuard", "[RefCore]") {
  SimpleEDProductGetter outer;
  SimpleEDProductGetter inner;
  REQUIRE(currentGetter() == nullptr);

  SECTION("sets and resets the getter") {
    {
      edm::RefCoreStreamerGuard guard(&outer);
      REQUIRE(currentGetter() == &outer);
    }
    REQUIRE(currentGetter() == nullptr);
  }

  SECTION("nested guards restore the outer getter") {
    {
      edm::RefCoreStreamerGuard guard(&outer);
      {
        edm::RefCoreStreamerGuard nested(&inner);
        REQUIRE(currentGetter() == &inner);
      }
      REQUIRE(currentGetter() == &outer);
    }
    REQUIRE(currentGetter() == nullptr);
  }

  SECTION("a nested null getter is used, then the outer getter is restored") {
    edm::RefCoreStreamerGuard guard(&outer);
    {
      edm::RefCoreStreamerGuard nested(nullptr);
      REQUIRE(currentGetter() == nullptr);
    }
    REQUIRE(currentGetter() == &outer);
  }
}
