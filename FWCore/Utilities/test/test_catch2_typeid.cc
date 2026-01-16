/*----------------------------------------------------------------------

Test program for edm::TypeID class.
Changed by Viji on 29-06-2005

 ----------------------------------------------------------------------*/

#include <cassert>
#include <iostream>
#include <string>
#include <catch2/catch_all.hpp>
#include "FWCore/Utilities/interface/TypeID.h"

namespace edmtest {
  struct empty {};
}  // namespace edmtest

TEST_CASE("TypeID", "[TypeID]") {
  SECTION("equalityTest") {
    edmtest::empty e;
    edm::TypeID id1(e);
    edm::TypeID id2(e);
    REQUIRE(!(id1 < id2));
    REQUIRE(!(id2 < id1));
    std::string n1(id1.name());
    std::string n2(id2.name());
    REQUIRE(n1 == n2);
  }

  SECTION("copyTest") {
    edmtest::empty e;
    edm::TypeID id1(e);
    edm::TypeID id3 = id1;
    REQUIRE(!(id1 < id3));
    REQUIRE(!(id3 < id1));
    std::string n1(id1.name());
    std::string n3(id3.name());
    REQUIRE(n1 == n3);
  }
}
