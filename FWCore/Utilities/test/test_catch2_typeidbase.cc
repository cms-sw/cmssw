/*----------------------------------------------------------------------

Test program for edm::TypeIDBase class.
Changed by Viji on 29-06-2005

 ----------------------------------------------------------------------*/

#include <cassert>
#include <iostream>
#include <string>
#include <catch2/catch_all.hpp>
#include "FWCore/Utilities/interface/TypeIDBase.h"

namespace edmtest {
  struct empty {};
}  // namespace edmtest

TEST_CASE("TypeIDBase", "[TypeIDBase]") {
  SECTION("equalityTest") {
    edmtest::empty e;
    edm::TypeIDBase id1(typeid(e));
    edm::TypeIDBase id2(typeid(e));
    REQUIRE(!(id1 < id2));
    REQUIRE(!(id2 < id1));
    std::string n1(id1.name());
    std::string n2(id2.name());
    REQUIRE(n1 == n2);
  }

  SECTION("copyTest") {
    edmtest::empty e;
    edm::TypeIDBase id1(typeid(e));
    edm::TypeIDBase id3 = id1;
    REQUIRE(!(id1 < id3));
    REQUIRE(!(id3 < id1));
    std::string n1(id1.name());
    std::string n3(id3.name());
    REQUIRE(n1 == n3);
  }
}
