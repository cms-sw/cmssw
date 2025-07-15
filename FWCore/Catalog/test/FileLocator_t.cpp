#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

#include "TestSiteLocalConfig.h"

#include <filesystem>
#include <string>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("FileLocator with Rucio data catalog", "[FWCore/Catalog]") {
  edm::ServiceToken tempToken = edmtest::catalog::makeTestSiteLocalConfigToken();

  SECTION("prefix") {
    edm::ServiceRegistry::Operate operate(tempToken);
    //empty catalog
    edm::CatalogAttributes tmp_cat;
    //use the first catalog provided by site local config
    edm::FileLocator fl(tmp_cat, 0);
    CHECK("root://cmsxrootd.fnal.gov/store/group/bha/bho" ==
          fl.pfn("/store/group/bha/bho", edm::CatalogType::RucioCatalog));
  }
  SECTION("rule") {
    edm::ServiceRegistry::Operate operate(tempToken);
    //empty catalog
    edm::CatalogAttributes tmp_cat;
    //use the second catalog provided by site local config
    edm::FileLocator fl(tmp_cat, 1);
    const std::array<const char*, 7> lfn = {{"/bha/bho",
                                             "bha",
                                             "file:bha",
                                             "file:/bha/bho",
                                             "/castor/cern.ch/cms/bha/bho",
                                             "someprotocol:/castor/cern.ch/cms/bha/bho",
                                             "someprotocol:/bha/bho"}};
    CHECK("root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/group/bha/bho" ==
          fl.pfn("/store/group/bha/bho", edm::CatalogType::RucioCatalog));
    for (auto file : lfn) {
      CHECK("" == fl.pfn(file, edm::CatalogType::RucioCatalog));
    }
  }
  SECTION("chainedrule") {
    edm::ServiceRegistry::Operate operate(tempToken);
    //empty catalog
    edm::CatalogAttributes tmp_cat;
    //use the third catalog provided by site local config above
    edm::FileLocator fl(tmp_cat, 2);
    const std::array<const char*, 7> lfn = {{"/bha/bho",
                                             "bha",
                                             "file:bha",
                                             "file:/bha/bho",
                                             "/castor/cern.ch/cms/bha/bho",
                                             "someprotocol:/castor/cern.ch/cms/bha/bho",
                                             "someprotocol:/bha/bho"}};
    //one level chain between "root" and "second" protocols (see storage.json)
    CHECK("root://host.domain//pnfs/cms/store/group/bha/bho" ==
          fl.pfn("/store/group/bha/bho", edm::CatalogType::RucioCatalog));
    //two levels chain between "root", "second" and "first" (see storage.json)
    CHECK("root://host.domain//pnfs/cms/store/user/AAA/bho" ==
          fl.pfn("/store/user/aaa/bho", edm::CatalogType::RucioCatalog));
    for (auto file : lfn) {
      CHECK("" == fl.pfn(file, edm::CatalogType::RucioCatalog));
    }
  }
}

TEST_CASE("FileLocator with TrivialFileCatalog", "[FWCore/Catalog]") {
  std::string CMSSW_BASE(std::getenv("CMSSW_BASE"));
  std::string CMSSW_RELEASE_BASE(std::getenv("CMSSW_RELEASE_BASE"));
  std::string file_name("/src/FWCore/Catalog/test/simple_catalog.xml");
  std::string full_file_name = std::filesystem::exists((CMSSW_BASE + file_name).c_str())
                                   ? CMSSW_BASE + file_name
                                   : CMSSW_RELEASE_BASE + file_name;

  //create the services
  std::vector<std::string> tmp{std::string("trivialcatalog_file:") + full_file_name + "?protocol=xrd"};
  edm::ServiceToken tempToken(edm::ServiceRegistry::createContaining(
      std::unique_ptr<edm::SiteLocalConfig>(std::make_unique<edmtest::catalog::TestSiteLocalConfig>(tmp))));

  //make the services available
  SECTION("standard") {
    edm::ServiceRegistry::Operate operate(tempToken);
    edm::FileLocator fl("");

    const std::array<const char*, 7> lfn = {{"/bha/bho",
                                             "bha",
                                             "file:bha",
                                             "file:/bha/bho",
                                             "/castor/cern.ch/cms/bha/bho",
                                             "someprotocol:/castor/cern.ch/cms/bha/bho",
                                             "someprotocol:/bha/bho"}};

    CHECK("/storage/path/store/group/bha/bho" == fl.pfn("/store/group/bha/bho", edm::CatalogType::TrivialCatalog));
    for (auto file : lfn) {
      CHECK("" == fl.pfn(file, edm::CatalogType::TrivialCatalog));
    }
  }

  SECTION("override") {
    edm::ServiceRegistry::Operate operate(tempToken);

    std::string override_file_name("/src/FWCore/Catalog/test/override_catalog.xml");
    std::string override_full_file_name = std::filesystem::exists((CMSSW_BASE + override_file_name).c_str())
                                              ? CMSSW_BASE + override_file_name
                                              : CMSSW_RELEASE_BASE + override_file_name;

    edm::FileLocator fl(("trivialcatalog_file:" + override_full_file_name + "?protocol=override").c_str());

    std::array<const char*, 8> lfn = {{"/store/group/bha/bho",
                                       "/bha/bho",
                                       "bha",
                                       "file:bha",
                                       "file:/bha/bho",
                                       "/castor/cern.ch/cms/bha/bho",
                                       "someprotocol:/castor/cern.ch/cms/bha/bho",
                                       "someprotocol:/bha/bho"}};

    auto const overriden_file =
        "/store/unmerged/relval/CMSSW_3_8_0_pre3/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V2-v1/0666/"
        "80EC0BCD-D279-DF11-B1DB-0030487C90EE.root";

    CHECK("/FULL_PATH_TO_THE_FIRST_STEP_ROOT_FILE/80EC0BCD-D279-DF11-B1DB-0030487C90EE.root" ==
          fl.pfn(overriden_file, edm::CatalogType::TrivialCatalog));

    for (auto f : lfn) {
      CHECK("" == fl.pfn(f, edm::CatalogType::TrivialCatalog));
    }
  }
}
