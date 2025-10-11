#include "catch2/catch_all.hpp"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/processingOrderMerge.h"
#include "FWCore/Utilities/interface/Exception.h"

TEST_CASE("processingOrderMerge", "[processingOrderMerge]") {
  SECTION("merge") {
    SECTION("empty vector") {
      SECTION("empty history") {
        edm::ProcessHistory ph;
        std::vector<std::string> names;
        edm::processingOrderMerge(ph, names);
        REQUIRE(names.empty());
      }
      SECTION("one process") {
        edm::ProcessHistory ph;
        edm::ProcessConfiguration pc("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.push_back(pc);
        std::vector<std::string> names;
        edm::processingOrderMerge(ph, names);
        REQUIRE(names.size() == 1);
        REQUIRE(names[0] == "A");
      }
      SECTION("two processes") {
        edm::ProcessHistory ph;
        ph.emplace_back("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("B", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names;
        edm::processingOrderMerge(ph, names);
        REQUIRE(names.size() == 2);
        REQUIRE(names[0] == "B");
        REQUIRE(names[1] == "A");
      }
    }
    SECTION("identical") {
      edm::ProcessHistory ph;
      ph.emplace_back("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
      ph.emplace_back("B", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
      std::vector<std::string> names = {"B", "A"};
      edm::processingOrderMerge(ph, names);
      REQUIRE(names.size() == 2);
      REQUIRE(names[0] == "B");
      REQUIRE(names[1] == "A");
    }
    SECTION("missing items from one") {
      SECTION("vector missing most recent from history") {
        edm::ProcessHistory ph;
        ph.emplace_back("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("B", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("C", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"B", "A"};
        edm::processingOrderMerge(ph, names);
        REQUIRE(names.size() == 3);
        REQUIRE(names[0] == "C");
        REQUIRE(names[1] == "B");
        REQUIRE(names[2] == "A");
      }
      SECTION("history missing most recent from vector") {
        edm::ProcessHistory ph;
        ph.emplace_back("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("B", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"C", "B", "A"};
        edm::processingOrderMerge(ph, names);
        REQUIRE(names.size() == 3);
        REQUIRE(names[0] == "C");
        REQUIRE(names[1] == "B");
        REQUIRE(names[2] == "A");
      }
      SECTION("vector missing last from history") {
        edm::ProcessHistory ph;
        ph.emplace_back("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("B", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("C", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"C", "B"};
        edm::processingOrderMerge(ph, names);
        REQUIRE(names.size() == 3);
        REQUIRE(names[0] == "C");
        REQUIRE(names[1] == "B");
        REQUIRE(names[2] == "A");
      }
      SECTION("history missing last from vector") {
        edm::ProcessHistory ph;
        ph.emplace_back("B", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("C", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"C", "B", "A"};
        edm::processingOrderMerge(ph, names);
        REQUIRE(names.size() == 3);
        REQUIRE(names[0] == "C");
        REQUIRE(names[1] == "B");
        REQUIRE(names[2] == "A");
      }
      SECTION("vector missing middle from history") {
        edm::ProcessHistory ph;
        ph.emplace_back("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("B", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("C", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"C", "A"};
        edm::processingOrderMerge(ph, names);
        REQUIRE(names.size() == 3);
        REQUIRE(names[0] == "C");
        REQUIRE(names[1] == "B");
        REQUIRE(names[2] == "A");
      }
      SECTION("history missing middle from vector") {
        edm::ProcessHistory ph;
        ph.emplace_back("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("C", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"C", "B", "A"};
        edm::processingOrderMerge(ph, names);
        REQUIRE(names.size() == 3);
        REQUIRE(names[0] == "C");
        REQUIRE(names[1] == "B");
        REQUIRE(names[2] == "A");
      }
    }
    SECTION("incompatible") {
      SECTION("independent 1") {
        edm::ProcessHistory ph;
        ph.emplace_back("C", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"B"};
        REQUIRE_THROWS_AS(edm::processingOrderMerge(ph, names), cms::Exception);
      }
      SECTION("independent 2") {
        edm::ProcessHistory ph;
        ph.emplace_back("C", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("D", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"B", "A"};
        REQUIRE_THROWS_AS(edm::processingOrderMerge(ph, names), cms::Exception);
      }
      SECTION("reversed order") {
        edm::ProcessHistory ph;
        ph.emplace_back("B", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"B", "A"};
        REQUIRE_THROWS_AS(edm::processingOrderMerge(ph, names), cms::Exception);
      }
      SECTION("longer reversed order") {
        edm::ProcessHistory ph;
        ph.emplace_back("C", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("B", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"C", "B", "A"};
        REQUIRE_THROWS_AS(edm::processingOrderMerge(ph, names), cms::Exception);
      }
      SECTION("ending inconsistent") {
        edm::ProcessHistory ph;
        ph.emplace_back("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("B", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("C", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"B", "C", "A"};
        REQUIRE_THROWS_AS(edm::processingOrderMerge(ph, names), cms::Exception);
      }
      SECTION("beginning inconsistent") {
        edm::ProcessHistory ph;
        ph.emplace_back("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("B", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("C", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"C", "A", "B"};
        REQUIRE_THROWS_AS(edm::processingOrderMerge(ph, names), cms::Exception);
      }
      SECTION("middle inconsistent") {
        edm::ProcessHistory ph;
        ph.emplace_back("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("B", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("C", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("D", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"D", "B", "C", "A"};
        REQUIRE_THROWS_AS(edm::processingOrderMerge(ph, names), cms::Exception);
      }
      SECTION("reverse order with an extra one in front") {
        edm::ProcessHistory ph;
        ph.emplace_back("C", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        ph.emplace_back("A", edm::ReleaseVersion(""), edm::HardwareResourcesDescription());
        std::vector<std::string> names = {"B", "C", "A"};
        REQUIRE_THROWS_AS(edm::processingOrderMerge(ph, names), cms::Exception);
      }
    }
  }
}
