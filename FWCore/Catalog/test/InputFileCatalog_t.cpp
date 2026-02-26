#include <string>
#include <vector>

#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Catalog/interface/StorageURLModifier.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include "TestSiteLocalConfig.h"
#include "TestScitagConfig.h"

#include "catch2/catch_all.hpp"

TEST_CASE("InputFileCatalog with Rucio data catalog", "[FWCore/Catalog]") {
  edm::ServiceToken tempToken = edmtest::catalog::makeTestSiteLocalConfigToken();
  edm::ServiceToken testToken = edmtest::catalog::makeTestScitagConfigToken(tempToken);

  SECTION("Empty") {
    edm::ServiceRegistry::Operate operate(tempToken);
    edm::InputFileCatalog catalog({}, "");
    REQUIRE(catalog.empty());
  }

  SECTION("isPhysical") {
    REQUIRE(edm::InputFileCatalog::isPhysical(""));
    REQUIRE(not edm::InputFileCatalog::isPhysical("/foo/bar"));
    REQUIRE(edm::InputFileCatalog::isPhysical("file:/foo/bar"));
  }

  // NOTE: This file is testing InputFileCatalog and it is not intended
  // to test the ScitagConfig service itself (there is a separate unit test
  // for that elsewhere). It is using a test replacement for the ScitagConfig
  // service NOT the actual ScitagConfig service.

  SECTION("Default behavior") {
    edm::ServiceRegistry::Operate operate(testToken);

    // Default SciTagCategory is Primary
    edm::InputFileCatalog catalog(std::vector<std::string>{"/store/foo/bar", "   file:/foo/bar", "root://foobar "}, "");
    REQUIRE(!catalog.empty());

    SECTION("fileNames") {
      SECTION("Catalog 0") {
        auto const names = catalog.fileNames(0);
        REQUIRE(names.size() == 3);
        CHECK(names[0] == "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=196664");
        CHECK(names[1] == "file:/foo/bar");
        CHECK(names[2] == "root://foobar?scitag.flow=196664");
      }
      // The fileNames() works only for catalog 0
      // Catalog 1 or 2 leads to a crash because the input file list
      // is a mixture of LFNs and PFNs
      // This isn't really good behavior...
    }

    SECTION("fileCatalogItems") {
      auto const& items = catalog.fileCatalogItems();
      REQUIRE(items.size() == 3);
      CHECK(items[0].logicalFileName() == "/store/foo/bar");
      CHECK(items[1].logicalFileName() == "");
      CHECK(items[2].logicalFileName() == "");

      REQUIRE(items[0].fileNames().size() == 3);
      REQUIRE(items[1].fileNames().size() == 1);
      REQUIRE(items[2].fileNames().size() == 1);

      SECTION("Catalog 0") {
        CHECK(items[0].fileName(0) == "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=196664");
        CHECK(items[1].fileName(0) == "file:/foo/bar");
        CHECK(items[2].fileName(0) == "root://foobar?scitag.flow=196664");
      }
      SECTION("Catalog 1") {
        CHECK(items[0].fileName(1) == "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar?scitag.flow=196664");
        CHECK(items[0].fileNames()[1] ==
              "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar?scitag.flow=196664");
      }
      SECTION("Catalog 2") {
        CHECK(items[0].fileName(2) == "root://host.domain//pnfs/cms/store/foo/bar?scitag.flow=196664");
      }
    }
  }

  SECTION("Embedded behavior") {
    edm::ServiceRegistry::Operate operate(testToken);

    edm::InputFileCatalog catalog(std::vector<std::string>{"   file:/foo/bar", "/store/foo/bar", "root://foobar "},
                                  "",
                                  false,
                                  edm::SciTagCategory::Embedded);
    REQUIRE(!catalog.empty());

    SECTION("fileCatalogItems") {
      auto const& items = catalog.fileCatalogItems();
      SECTION("Embedded Catalog 0") {
        CHECK(items[0].fileName(0) == "file:/foo/bar");
        CHECK(items[1].fileName(0) == "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=196700");
        CHECK(items[2].fileName(0) == "root://foobar?scitag.flow=196700");
      }
      SECTION("Embedded Catalog 1") {
        CHECK(items[1].fileName(1) == "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar?scitag.flow=196700");
      }
      SECTION("Embedded Catalog 2") {
        CHECK(items[1].fileName(2) == "root://host.domain//pnfs/cms/store/foo/bar?scitag.flow=196700");
      }
    }
  }

  SECTION("PreMixedPileup behavior") {
    edm::ServiceRegistry::Operate operate(testToken);

    edm::InputFileCatalog catalog(std::vector<std::string>{"   file:/foo/bar", "/store/foo/bar", "root://foobar "},
                                  "",
                                  false,
                                  edm::SciTagCategory::PreMixedPileup);
    auto const& items = catalog.fileCatalogItems();
    CHECK(items[1].fileName(0) == "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=196704");
  }

  SECTION("Behavior of 'Undefined' category") {
    edm::ServiceRegistry::Operate operate(tempToken);

    edm::InputFileCatalog catalog(std::vector<std::string>{"   file:/foo/bar", "/store/foo/bar", "root://foobar "},
                                  "",
                                  false,
                                  edm::SciTagCategory::Undefined);
    auto const& items = catalog.fileCatalogItems();
    CHECK(items[1].fileName(0) == "root://cmsxrootd.fnal.gov/store/foo/bar");
  }

  SECTION("Override catalog") {
    edm::ServiceRegistry::Operate operate(testToken);

    edm::InputFileCatalog catalog(std::vector<std::string>{"/store/foo/bar", "   file:/foo/bar", "root://foobar "},
                                  "T1_US_FNAL,,T1_US_FNAL,FNAL_dCache_EOS,XRootD");

    SECTION("fileNames") {
      auto const names = catalog.fileNames(0);
      REQUIRE(names.size() == 3);
      CHECK(names[0] == "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar?scitag.flow=196664");
      CHECK(names[1] == "file:/foo/bar");
      CHECK(names[2] == "root://foobar?scitag.flow=196664");
    }

    SECTION("fileCatalogItems") {
      auto const& items = catalog.fileCatalogItems();
      REQUIRE(items.size() == 3);

      CHECK(items[0].logicalFileName() == "/store/foo/bar");
      CHECK(items[1].logicalFileName() == "");
      CHECK(items[2].logicalFileName() == "");

      REQUIRE(items[0].fileNames().size() == 1);
      REQUIRE(items[1].fileNames().size() == 1);
      REQUIRE(items[2].fileNames().size() == 1);

      CHECK(items[0].fileName(0) == "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar?scitag.flow=196664");
      CHECK(items[1].fileName(0) == "file:/foo/bar");
      CHECK(items[2].fileName(0) == "root://foobar?scitag.flow=196664");
    }
  }

  SECTION("useLFNasPFNifLFNnotFound") {
    edm::ServiceRegistry::Operate operate(testToken);

    bool useLFNasPFNifLFNnotFound = true;
    edm::InputFileCatalog catalog(
        std::vector<std::string>{"/store/foo/bar", "/tmp/foo/bar", "root://foobar "}, "", useLFNasPFNifLFNnotFound);

    SECTION("fileNames") {
      SECTION("Catalog 0") {
        auto const names = catalog.fileNames(0);
        REQUIRE(names.size() == 3);
        CHECK(names[0] == "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=196664");
        CHECK(names[1] == "/tmp/foo/bar");
        CHECK(names[2] == "root://foobar?scitag.flow=196664");
      }
    }

    SECTION("fileCatalogItems") {
      auto const& items = catalog.fileCatalogItems();
      REQUIRE(items.size() == 3);
      CHECK(items[0].logicalFileName() == "/store/foo/bar");
      CHECK(items[1].logicalFileName() == "/tmp/foo/bar");
      CHECK(items[2].logicalFileName() == "");

      REQUIRE(items[0].fileNames().size() == 3);
      REQUIRE(items[1].fileNames().size() == 3);
      REQUIRE(items[2].fileNames().size() == 1);

      SECTION("Catalog 0") {
        CHECK(items[0].fileName(0) == "root://cmsxrootd.fnal.gov/store/foo/bar?scitag.flow=196664");
        CHECK(items[1].fileName(0) == "/tmp/foo/bar");
        CHECK(items[2].fileName(0) == "root://foobar?scitag.flow=196664");
      }
      SECTION("Catalog 1") {
        CHECK(items[0].fileName(1) == "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar?scitag.flow=196664");
        CHECK(items[1].fileName(1) == "/tmp/foo/bar");
      }
      SECTION("Catalog 2") {
        CHECK(items[0].fileName(2) == "root://host.domain//pnfs/cms/store/foo/bar?scitag.flow=196664");
        CHECK(items[1].fileName(2) == "/tmp/foo/bar");
      }
    }
  }
}
