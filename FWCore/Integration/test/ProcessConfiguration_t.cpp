#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cassert>
#include <iostream>
#include <string>

TEST_CASE("test ProcessConfiguration", "[ProcessConfiguration]") {
  edm::ParameterSet dummyPset;
  dummyPset.registerIt();
  edm::ParameterSetID id = dummyPset.id();
  SECTION("default initialize") {
    SECTION("self operator==") {
      edm::ProcessConfiguration pc1;
      pc1.setParameterSetID(id);
      REQUIRE(pc1 == pc1);
    }
    SECTION("equivalence") {
      edm::ProcessConfiguration pc1;
      edm::ProcessConfiguration pc2;
      pc1.setParameterSetID(id);
      pc2.setParameterSetID(id);
      REQUIRE(pc1 == pc2);
    }
  }
  SECTION("non-default initialized") {
    edm::ProcessConfiguration pc1;
    edm::ProcessConfiguration pc2("reco2", edm::ParameterSetID(), std::string(), std::string());
    edm::ProcessConfiguration pc3("reco3", edm::ParameterSetID(), std::string(), std::string());
    pc1.setParameterSetID(id);
    pc2.setParameterSetID(id);
    pc3.setParameterSetID(id);
    edm::ProcessConfigurationID id1 = pc1.id();
    edm::ProcessConfigurationID id2 = pc2.id();
    edm::ProcessConfigurationID id3 = pc3.id();

    SECTION("non-equivalence") {
      REQUIRE(id1 != id2);
      REQUIRE(id2 != id3);
      REQUIRE(id3 != id1);
    }

    SECTION("equivalence") {
      edm::ProcessConfiguration pc4("reco2", edm::ParameterSetID(), std::string(), std::string());
      pc4.setParameterSetID(id);
      edm::ProcessConfigurationID id4 = pc4.id();
      REQUIRE(pc4 == pc2);
      REQUIRE(id4 == id2);
    }
  }
}
