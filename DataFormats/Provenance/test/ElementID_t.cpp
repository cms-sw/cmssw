#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "DataFormats/Provenance/interface/ElementID.h"

#include <sstream>

TEST_CASE("ElementID", "[ElementID]") {
  SECTION("Default construction is invalid") { REQUIRE(edm::ElementID{}.isValid() == false); }

  SECTION("Basic operations") {
    edm::ElementID id{edm::ProductID{1, 2}, 3};
    REQUIRE(id.isValid() == true);
    REQUIRE(id.id() == edm::ProductID{1, 2});
    REQUIRE(id.key() == 3);
    REQUIRE(id.index() == 3);

    edm::ElementID id2;
    edm::swap(id, id2);
    REQUIRE(id.isValid() == false);
    REQUIRE(id2.id() == edm::ProductID{1, 2});
    REQUIRE(id2.key() == 3);
    REQUIRE(id2.index() == 3);

    REQUIRE(id2 == edm::ElementID{edm::ProductID{1, 2}, 3});
    REQUIRE(id2 != edm::ElementID{edm::ProductID{2, 2}, 3});
    REQUIRE(id2 != edm::ElementID{edm::ProductID{1, 3}, 3});
    REQUIRE(id2 != edm::ElementID{edm::ProductID{1, 2}, 4});

    REQUIRE(id2 < edm::ElementID{edm::ProductID{1, 2}, 4});
    REQUIRE(id2 < edm::ElementID{edm::ProductID{1, 3}, 3});
    REQUIRE(id2 < edm::ElementID{edm::ProductID{2, 2}, 3});

    REQUIRE(not(id2 < id2));

    REQUIRE(edm::ElementID{edm::ProductID{1, 2}, 2} < id2);
    REQUIRE(edm::ElementID{edm::ProductID{1, 1}, 3} < id2);
    REQUIRE(edm::ElementID{edm::ProductID{0, 2}, 3} < id2);

    std::stringstream ss;
    ss << id2;
    REQUIRE(ss.str() == "1:2:3");

    id2.reset();
    REQUIRE(id2.isValid() == false);
  }
}
