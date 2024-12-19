#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

#include "TestSiteLocalConfig.h"

#include "catch.hpp"

TEST_CASE("InputFileCatalog with Rucio data catalog", "[FWCore/Catalog]") {
  edm::ServiceToken tempToken = edmtest::catalog::makeTestSiteLocalConfigToken();

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

  SECTION("Default behavior") {
    edm::ServiceRegistry::Operate operate(tempToken);

    edm::InputFileCatalog catalog(std::vector<std::string>{"/store/foo/bar", "   file:/foo/bar", "root://foobar "}, "");

    SECTION("logicalFileNames") {
      auto const& lfns = catalog.logicalFileNames();
      REQUIRE(lfns.size() == 3);
      CHECK(lfns[0] == "/store/foo/bar");
      CHECK(lfns[1] == "");  // was PFN
      CHECK(lfns[2] == "");  // was PFN
    }

    SECTION("fileNames") {
      SECTION("Catalog 0") {
        auto const names = catalog.fileNames(0);
        REQUIRE(names.size() == 3);
        CHECK(names[0] == "root://cmsxrootd.fnal.gov/store/foo/bar");
        CHECK(names[1] == "file:/foo/bar");
        CHECK(names[2] == "root://foobar");
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
        CHECK(items[0].fileName(0) == "root://cmsxrootd.fnal.gov/store/foo/bar");
        CHECK(items[1].fileName(0) == "file:/foo/bar");
        CHECK(items[2].fileName(0) == "root://foobar");
      }
      SECTION("Catalog 1") {
        CHECK(items[0].fileName(1) == "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar");
      }
      SECTION("Catalog 2") { CHECK(items[0].fileName(2) == "root://host.domain//pnfs/cms/store/foo/bar"); }
    }
  }

  SECTION("Override catalog") {
    edm::ServiceRegistry::Operate operate(tempToken);

    edm::InputFileCatalog catalog(std::vector<std::string>{"/store/foo/bar", "   file:/foo/bar", "root://foobar "},
                                  "T1_US_FNAL,,T1_US_FNAL,FNAL_dCache_EOS,XRootD");

    SECTION("logicalFileNames") {
      auto const& lfns = catalog.logicalFileNames();
      REQUIRE(lfns.size() == 3);
      CHECK(lfns[0] == "/store/foo/bar");
      CHECK(lfns[1] == "");  // was PFN
      CHECK(lfns[2] == "");  // was PFN
    }

    SECTION("fileNames") {
      auto const names = catalog.fileNames(0);
      REQUIRE(names.size() == 3);
      CHECK(names[0] == "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar");
      CHECK(names[1] == "file:/foo/bar");
      CHECK(names[2] == "root://foobar");
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

      CHECK(items[0].fileName(0) == "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar");
      CHECK(items[1].fileName(0) == "file:/foo/bar");
      CHECK(items[2].fileName(0) == "root://foobar");
    }
  }

  SECTION("useLFNasPFNifLFNnotFound") {
    edm::ServiceRegistry::Operate operate(tempToken);

    edm::InputFileCatalog catalog(
        std::vector<std::string>{"/store/foo/bar", "/tmp/foo/bar", "root://foobar "}, "", true);

    SECTION("logicalFileNames") {
      auto const& lfns = catalog.logicalFileNames();
      REQUIRE(lfns.size() == 3);
      CHECK(lfns[0] == "/store/foo/bar");
      CHECK(lfns[1] == "/tmp/foo/bar");
      CHECK(lfns[2] == "");  // was PFN
    }

    SECTION("fileNames") {
      SECTION("Catalog 0") {
        auto const names = catalog.fileNames(0);
        REQUIRE(names.size() == 3);
        CHECK(names[0] == "root://cmsxrootd.fnal.gov/store/foo/bar");
        CHECK(names[1] == "/tmp/foo/bar");
        CHECK(names[2] == "root://foobar");
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
        CHECK(items[0].fileName(0) == "root://cmsxrootd.fnal.gov/store/foo/bar");
        CHECK(items[1].fileName(0) == "/tmp/foo/bar");
        CHECK(items[2].fileName(0) == "root://foobar");
      }
      SECTION("Catalog 1") {
        CHECK(items[0].fileName(1) == "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/foo/bar");
        CHECK(items[1].fileName(1) == "/tmp/foo/bar");
      }
      SECTION("Catalog 2") {
        CHECK(items[0].fileName(2) == "root://host.domain//pnfs/cms/store/foo/bar");
        CHECK(items[1].fileName(2) == "/tmp/foo/bar");
      }
    }
  }
}
