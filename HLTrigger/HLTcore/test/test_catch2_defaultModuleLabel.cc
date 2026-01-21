#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include "catch2/catch_all.hpp"

class Bar {};
class MultiWordName {};
class SHOUTING {};
class ABCDef {};
template <typename T>
class Templated {};

namespace for_testing {
  class Foo {};
  template <typename T>
  class Templated2 {};
}  // namespace for_testing

namespace io_v1 {
  class Fii {};
}  // namespace io_v1

using Fii = io_v1::Fii;

namespace reco {
  namespace io_v8 {
    class Fii {};
  }  // namespace io_v8
  using Fii = io_v8::Fii;
}  // namespace reco

TEST_CASE("Test defaultModuleLabel", "[defaultModuleLabel]") {
  SECTION("no namespace") {
    CHECK(defaultModuleLabel<Bar>() == "bar");
    CHECK(defaultModuleLabel<MultiWordName>() == "multiWordName");
    CHECK(defaultModuleLabel<SHOUTING>() == "shouting");
    CHECK(defaultModuleLabel<ABCDef>() == "abcDef");
    SECTION("templated") {
      CHECK(defaultModuleLabel<Templated<Bar>>() == "templatedBar");
      CHECK(defaultModuleLabel<Templated<MultiWordName>>() == "templatedMultiWordName");
      CHECK(defaultModuleLabel<Templated<SHOUTING>>() == "templatedSHOUTING");
      CHECK(defaultModuleLabel<Templated<ABCDef>>() == "templatedABCDef");
    }
  }
  SECTION("with namespace") {
    CHECK(defaultModuleLabel<for_testing::Foo>() == "forTestingFoo");
    SECTION("templated") {
      CHECK(defaultModuleLabel<for_testing::Templated2<Bar>>() == "forTestingTemplated2Bar");
      CHECK(defaultModuleLabel<for_testing::Templated2<MultiWordName>>() == "forTestingTemplated2MultiWordName");
      CHECK(defaultModuleLabel<for_testing::Templated2<SHOUTING>>() == "forTestingTemplated2SHOUTING");
      CHECK(defaultModuleLabel<for_testing::Templated2<ABCDef>>() == "forTestingTemplated2ABCDef");
    }
  }
  SECTION("with version namespace") {
    CHECK(defaultModuleLabel<Fii>() == "fii");
    CHECK(defaultModuleLabel<reco::Fii>() == "recoFii");
    SECTION("templated") {
      CHECK(defaultModuleLabel<for_testing::Templated2<Fii>>() == "forTestingTemplated2Fii");
      CHECK(defaultModuleLabel<for_testing::Templated2<reco::Fii>>() == "forTestingTemplated2RecoFii");
    }
  }
}
