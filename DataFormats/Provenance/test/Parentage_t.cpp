#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cassert>
#include <iostream>

#include "catch.hpp"

TEST_CASE("test Parentage", "[Parentage]") {
  edm::Parentage ed1;
  CHECK(ed1 == ed1);
  edm::Parentage ed2;
  CHECK(ed1 == ed2);

  ed2.setParents(std::vector<edm::BranchID>(1));
  edm::Parentage ed3;
  ed3.setParents(std::vector<edm::BranchID>(2));

  edm::ParentageID id1 = ed1.id();
  edm::ParentageID id2 = ed2.id();
  edm::ParentageID id3 = ed3.id();

  CHECK(id1 != id2);
  CHECK(ed1 != ed2);
  CHECK(id1 != id3);
  CHECK(ed1 != ed3);
  CHECK(id2 != id3);
  CHECK(ed2 != ed3);

  edm::Parentage ed4;
  ed4.setParents(std::vector<edm::BranchID>(1));
  edm::ParentageID id4 = ed4.id();
  CHECK(ed4 == ed2);
  CHECK(id4 == id2);

  SECTION("ParentageID unchanging") {
    {
      const std::string idString = "d41d8cd98f00b204e9800998ecf8427e";
      std::string toString;
      id1.toString(toString);
      CHECK(toString == idString);
    }

    {
      const std::string idString = "2e5751b7cfd7f053cd29e946fb2649a4";
      std::string toString;
      id2.toString(toString);
      CHECK(toString == idString);
    }
    {
      const std::string idString = "20e13ca818af45e50e369e50db3914b8";
      std::string toString;
      id3.toString(toString);
      CHECK(toString == idString);
    }
    {
      edm::Parentage ed_mult;
      ed_mult.setParents(std::vector<edm::BranchID>({edm::BranchID(1), edm::BranchID(2), edm::BranchID(3)}));
      auto id_mult = ed_mult.id();
      const std::string idString = "6a5cf1697e50ec8e8dbe7a28ccad348b";
      std::string toString;
      id_mult.toString(toString);
      CHECK(toString == idString);
    }
  }
}
