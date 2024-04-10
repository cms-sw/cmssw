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
}
