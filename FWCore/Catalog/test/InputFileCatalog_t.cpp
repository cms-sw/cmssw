#include <string>
#include <vector>

#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include "TestSiteLocalConfig.h"

#include "catch.hpp"

TEST_CASE("InputFileCatalog with Rucio data catalog", "[FWCore/Catalog]") {
  edm::ServiceToken tempToken = edmtest::catalog::makeTestSiteLocalConfigToken();

  SECTION("Empty") {
    edm::ServiceRegistry::Operate operate(tempToken);
    edm::InputFileCatalog catalog(std::vector<std::string>{}, "");
    REQUIRE(catalog.empty());
    REQUIRE(catalog.configuredFileNames().empty());
  }

  SECTION("isPhysicalFileName") {
    REQUIRE(not edm::InputFileCatalog::isPhysicalFileName("/foo/bar"));
    REQUIRE(not edm::InputFileCatalog::isPhysicalFileName(""));
    REQUIRE(edm::InputFileCatalog::isPhysicalFileName("file:/foo/bar"));
  }

  SECTION("Default behavior") {
    edm::ServiceRegistry::Operate operate(tempToken);

    edm::InputFileCatalog catalog(std::vector<std::string>{"/store/foo/bar", "   file:/foo/bar", "root://foobar "}, "");

    SECTION("notEmpty") { REQUIRE(!catalog.empty()); }

    SECTION("logicalFileNames") {
      auto const& configuredFileNames = catalog.configuredFileNames();
      REQUIRE(configuredFileNames.size() == 3);
      CHECK(configuredFileNames[0] == "/store/foo/bar");
      CHECK(configuredFileNames[1] == "file:/foo/bar");
      CHECK(configuredFileNames[2] == "root://foobar");
    }

    SECTION("physicalFileNames") {
      auto const& configuredFileNames = catalog.configuredFileNames();

      REQUIRE(catalog.physicalFileNames(configuredFileNames[0]).size() == 3);
      REQUIRE(catalog.physicalFileNames(configuredFileNames[1]).size() == 1);
      REQUIRE(catalog.physicalFileNames(configuredFileNames[2]).size() == 1);

      SECTION("Catalog 0") {
        CHECK(catalog.physicalFileNames(configuredFileNames[0])[0] == "root://cmsxrootd.fnal.gov/store/foo/bar");
        CHECK(catalog.physicalFileNames(configuredFileNames[1])[0] == "file:/foo/bar");
        CHECK(catalog.physicalFileNames(configuredFileNames[2])[0] == "root://foobar");
      }
      SECTION("Catalog 1") {
        CHECK(catalog.physicalFileNames(configuredFileNames[0])[1] ==
              "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar");
      }
      SECTION("Catalog 2") {
        CHECK(catalog.physicalFileNames(configuredFileNames[0])[2] == "root://host.domain//pnfs/cms/store/foo/bar");
      }
    }

    SECTION("physicalFileNamesUsingIndex") {
      REQUIRE(catalog.physicalFileNames(0).size() == 3);
      REQUIRE(catalog.physicalFileNames(1).size() == 1);
      REQUIRE(catalog.physicalFileNames(2).size() == 1);

      SECTION("Catalog 0") {
        CHECK(catalog.physicalFileNames(0)[0] == "root://cmsxrootd.fnal.gov/store/foo/bar");
        CHECK(catalog.physicalFileNames(1)[0] == "file:/foo/bar");
        CHECK(catalog.physicalFileNames(2)[0] == "root://foobar");
      }
      SECTION("Catalog 1") {
        CHECK(catalog.physicalFileNames(0)[1] == "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar");
      }
      SECTION("Catalog 2") { CHECK(catalog.physicalFileNames(0)[2] == "root://host.domain//pnfs/cms/store/foo/bar"); }
    }

    SECTION("allPFNsFromFirstCatalog") {
      auto const pfns = catalog.allPFNsFromFirstCatalog();
      REQUIRE(pfns.size() == 3);
      CHECK(pfns[0] == "root://cmsxrootd.fnal.gov/store/foo/bar");
      CHECK(pfns[1] == "file:/foo/bar");
      CHECK(pfns[2] == "root://foobar");
    }

    SECTION("firstPFNFromFirstCatalog") {
      auto const pfn = catalog.firstPFNFromFirstCatalog();
      CHECK(pfn == "root://cmsxrootd.fnal.gov/store/foo/bar");
    }
  }

  SECTION("Override catalog") {
    edm::ServiceRegistry::Operate operate(tempToken);

    edm::InputFileCatalog catalog(std::vector<std::string>{"/store/foo/bar", "   file:/foo/bar", "root://foobar "},
                                  "T1_US_FNAL,,T1_US_FNAL,FNAL_dCache_EOS,XRootD");

    SECTION("physicalFileNames") {
      auto const& configuredFileNames = catalog.configuredFileNames();
      REQUIRE(configuredFileNames.size() == 3);
      CHECK(configuredFileNames[0] == "/store/foo/bar");
      CHECK(configuredFileNames[1] == "file:/foo/bar");
      CHECK(configuredFileNames[2] == "root://foobar");

      REQUIRE(catalog.physicalFileNames(configuredFileNames[0]).size() == 1);
      REQUIRE(catalog.physicalFileNames(configuredFileNames[1]).size() == 1);
      REQUIRE(catalog.physicalFileNames(configuredFileNames[2]).size() == 1);

      CHECK(catalog.physicalFileNames(configuredFileNames[0])[0] ==
            "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar");
      CHECK(catalog.physicalFileNames(configuredFileNames[1])[0] == "file:/foo/bar");
      CHECK(catalog.physicalFileNames(configuredFileNames[2])[0] == "root://foobar");
    }

    SECTION("allPFNsFromFirstCatalog") {
      auto const pfns = catalog.allPFNsFromFirstCatalog();
      REQUIRE(pfns.size() == 3);
      CHECK(pfns[0] == "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar");
      CHECK(pfns[1] == "file:/foo/bar");
      CHECK(pfns[2] == "root://foobar");
    }
  }

  SECTION("useLFNasPFNifLFNnotFound") {
    edm::ServiceRegistry::Operate operate(tempToken);

    edm::InputFileCatalog catalog(
        std::vector<std::string>{"/store/foo/bar", "/tmp/foo/bar", "root://foobar "}, "", true);

    SECTION("physicalFileNames") {
      auto const& configuredFileNames = catalog.configuredFileNames();
      REQUIRE(configuredFileNames.size() == 3);
      CHECK(configuredFileNames[0] == "/store/foo/bar");
      CHECK(configuredFileNames[1] == "/tmp/foo/bar");
      CHECK(configuredFileNames[2] == "root://foobar");

      REQUIRE(catalog.physicalFileNames(configuredFileNames[0]).size() == 3);
      REQUIRE(catalog.physicalFileNames(configuredFileNames[1]).size() == 3);
      REQUIRE(catalog.physicalFileNames(configuredFileNames[2]).size() == 1);

      SECTION("Catalog 0") {
        CHECK(catalog.physicalFileNames(configuredFileNames[0])[0] == "root://cmsxrootd.fnal.gov/store/foo/bar");
        CHECK(catalog.physicalFileNames(configuredFileNames[1])[0] == "/tmp/foo/bar");
        CHECK(catalog.physicalFileNames(configuredFileNames[2])[0] == "root://foobar");
      }
      SECTION("Catalog 1") {
        CHECK(catalog.physicalFileNames(configuredFileNames[0])[1] ==
              "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar");
        CHECK(catalog.physicalFileNames(configuredFileNames[1])[1] == "/tmp/foo/bar");
      }
      SECTION("Catalog 2") {
        CHECK(catalog.physicalFileNames(configuredFileNames[0])[2] == "root://host.domain//pnfs/cms/store/foo/bar");
        CHECK(catalog.physicalFileNames(configuredFileNames[1])[2] == "/tmp/foo/bar");
      }
    }

    SECTION("allPFNsFromFirstCatalog") {
      SECTION("Catalog 0") {
        auto const pfns = catalog.allPFNsFromFirstCatalog();
        REQUIRE(pfns.size() == 3);
        CHECK(pfns[0] == "root://cmsxrootd.fnal.gov/store/foo/bar");
        CHECK(pfns[1] == "/tmp/foo/bar");
        CHECK(pfns[2] == "root://foobar");
      }
    }
  }
}
