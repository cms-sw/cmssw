#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWStorage/Catalog/interface/InputFileCatalog.h"
#include "FWStorage/Catalog/interface/StorageURLModifier.h"

#include "TestSiteLocalConfig.h"
#include "TestScitagConfig.h"

#include "catch2/catch_all.hpp"

TEST_CASE("InputFileCatalog with Rucio data catalog", "[FWStorage/Catalog]") {
  edm::ServiceToken tempToken = edmtest::catalog::makeTestSiteLocalConfigToken();
  edm::ServiceToken testToken = edmtest::catalog::makeTestScitagConfigToken(tempToken);

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

  // NOTE: This file is testing InputFileCatalog and it is not intended
  // to test the ScitagConfig service itself (there is a separate unit test
  // for that elsewhere). It is using a test replacement for the ScitagConfig
  // service NOT the actual ScitagConfig service.

  SECTION("Default behavior") {
    edm::ServiceRegistry::Operate operate(testToken);

    // Default SciTagCategory is Primary
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
        CHECK(catalog.physicalFileNames(configuredFileNames[0])[0] ==
              "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=206");
        CHECK(catalog.physicalFileNames(configuredFileNames[1])[0] == "file:/foo/bar");
        CHECK(catalog.physicalFileNames(configuredFileNames[2])[0] == "root://foobar?scitag.flow=206");
      }
      SECTION("Catalog 1") {
        CHECK(catalog.physicalFileNames(configuredFileNames[0])[1] ==
              "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar?scitag.flow=206");
      }
      SECTION("Catalog 2") {
        CHECK(catalog.physicalFileNames(configuredFileNames[0])[2] ==
              "root://host.domain//pnfs/cms/store/foo/bar?scitag.flow=206");
      }
    }

    SECTION("physicalFileNamesUsingIndex") {
      REQUIRE(catalog.physicalFileNames(0).size() == 3);
      REQUIRE(catalog.physicalFileNames(1).size() == 1);
      REQUIRE(catalog.physicalFileNames(2).size() == 1);

      SECTION("Catalog 0") {
        CHECK(catalog.physicalFileNames(0)[0] == "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=206");
        CHECK(catalog.physicalFileNames(1)[0] == "file:/foo/bar");
        CHECK(catalog.physicalFileNames(2)[0] == "root://foobar?scitag.flow=206");
      }
      SECTION("Catalog 1") {
        CHECK(catalog.physicalFileNames(0)[1] ==
              "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar?scitag.flow=206");
      }
      SECTION("Catalog 2") {
        CHECK(catalog.physicalFileNames(0)[2] == "root://host.domain//pnfs/cms/store/foo/bar?scitag.flow=206");
      }
    }

    SECTION("allPFNsFromFirstCatalog") {
      auto const pfns = catalog.allPFNsFromFirstCatalog();
      REQUIRE(pfns.size() == 3);
      CHECK(pfns[0] == "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=206");
      CHECK(pfns[1] == "file:/foo/bar");
      CHECK(pfns[2] == "root://foobar?scitag.flow=206");
    }

    SECTION("firstPFNFromFirstCatalog") {
      auto const pfn = catalog.firstPFNFromFirstCatalog();
      CHECK(pfn == "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=206");
    }
  }

  SECTION("Embedded behavior") {
    edm::ServiceRegistry::Operate operate(testToken);

    edm::InputFileCatalog catalog(std::vector<std::string>{"   file:/foo/bar", "/store/foo/bar", "root://foobar "},
                                  "",
                                  false,
                                  edm::SciTagCategory::Embedded);
    REQUIRE(!catalog.empty());
    auto const& configuredFileNames = catalog.configuredFileNames();
    REQUIRE(configuredFileNames.size() == 3);

    SECTION("physicalFileNames") {
      SECTION("Embedded Catalog 0") {
        CHECK(catalog.physicalFileNames(configuredFileNames[0])[0] == "file:/foo/bar");
        CHECK(catalog.physicalFileNames(configuredFileNames[1])[0] ==
              "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=215");
        CHECK(catalog.physicalFileNames(configuredFileNames[2])[0] == "root://foobar?scitag.flow=215");
      }
      SECTION("Embedded Catalog 1") {
        CHECK(catalog.physicalFileNames(configuredFileNames[1])[1] ==
              "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar?scitag.flow=215");
      }
      SECTION("Embedded Catalog 2") {
        CHECK(catalog.physicalFileNames(configuredFileNames[1])[2] ==
              "root://host.domain//pnfs/cms/store/foo/bar?scitag.flow=215");
      }
    }
  }

  SECTION("PreMixedPileup behavior") {
    edm::ServiceRegistry::Operate operate(testToken);

    edm::InputFileCatalog catalog(std::vector<std::string>{"   file:/foo/bar", "/store/foo/bar", "root://foobar "},
                                  "",
                                  false,
                                  edm::SciTagCategory::PreMixedPileup);
    auto const& configuredFileNames = catalog.configuredFileNames();
    REQUIRE(configuredFileNames.size() == 3);

    CHECK(catalog.physicalFileNames(configuredFileNames[1])[0] ==
          "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=216");
  }

  SECTION("Behavior of 'Undefined' category") {
    edm::ServiceRegistry::Operate operate(tempToken);

    edm::InputFileCatalog catalog(std::vector<std::string>{"   file:/foo/bar", "/store/foo/bar", "root://foobar "},
                                  "",
                                  false,
                                  edm::SciTagCategory::Undefined);
    auto const& configuredFileNames = catalog.configuredFileNames();
    REQUIRE(configuredFileNames.size() == 3);

    CHECK(catalog.physicalFileNames(configuredFileNames[1])[0] == "root://cmsxrootd.fnal.gov/store/foo/bar");
  }

  SECTION("Override catalog") {
    edm::ServiceRegistry::Operate operate(testToken);

    edm::ParameterSet pset;
    pset.addUntrackedParameter<std::vector<std::string>>(
        "fileNames", std::vector<std::string>{"/store/foo/bar", "   file:/foo/bar", "root://foobar "});
    pset.addUntrackedParameter<std::string>("overrideCatalog", "T1_US_FNAL,,T1_US_FNAL,FNAL_dCache_EOS,XRootD");
    edm::InputFileCatalog catalog(pset);

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
            "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar?scitag.flow=206");
      CHECK(catalog.physicalFileNames(configuredFileNames[1])[0] == "file:/foo/bar");
      CHECK(catalog.physicalFileNames(configuredFileNames[2])[0] == "root://foobar?scitag.flow=206");
    }

    SECTION("allPFNsFromFirstCatalog") {
      auto const pfns = catalog.allPFNsFromFirstCatalog();
      REQUIRE(pfns.size() == 3);
      CHECK(pfns[0] == "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar?scitag.flow=206");
      CHECK(pfns[1] == "file:/foo/bar");
      CHECK(pfns[2] == "root://foobar?scitag.flow=206");
    }
  }

  SECTION("useLFNasPFNifLFNnotFound") {
    edm::ServiceRegistry::Operate operate(testToken);

    bool useLFNasPFNifLFNnotFound = true;
    edm::InputFileCatalog catalog(
        std::vector<std::string>{"/store/foo/bar", "/tmp/foo/bar", "root://foobar "}, "", useLFNasPFNifLFNnotFound);

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
        CHECK(catalog.physicalFileNames(configuredFileNames[0])[0] ==
              "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=206");
        CHECK(catalog.physicalFileNames(configuredFileNames[1])[0] == "/tmp/foo/bar");
        CHECK(catalog.physicalFileNames(configuredFileNames[2])[0] == "root://foobar?scitag.flow=206");
      }
      SECTION("Catalog 1") {
        CHECK(catalog.physicalFileNames(configuredFileNames[0])[1] ==
              "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar?scitag.flow=206");
        CHECK(catalog.physicalFileNames(configuredFileNames[1])[1] == "/tmp/foo/bar");
      }
      SECTION("Catalog 2") {
        CHECK(catalog.physicalFileNames(configuredFileNames[0])[2] ==
              "root://host.domain//pnfs/cms/store/foo/bar?scitag.flow=206");
        CHECK(catalog.physicalFileNames(configuredFileNames[1])[2] == "/tmp/foo/bar");
      }
    }

    SECTION("allPFNsFromFirstCatalog") {
      SECTION("Catalog 0") {
        auto const pfns = catalog.allPFNsFromFirstCatalog();
        REQUIRE(pfns.size() == 3);
        CHECK(pfns[0] == "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=206");
        CHECK(pfns[1] == "/tmp/foo/bar");
        CHECK(pfns[2] == "root://foobar?scitag.flow=206");
      }
    }
  }
}
