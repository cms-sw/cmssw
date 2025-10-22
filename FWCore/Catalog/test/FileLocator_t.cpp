#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

#include <array>
#include <string>

#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include "TestSiteLocalConfig.h"

TEST_CASE("FileLocator with Rucio data catalog", "[FWCore/Catalog]") {
  edm::ServiceToken tempToken = edmtest::catalog::makeTestSiteLocalConfigToken();

  SECTION("prefix") {
    edm::ServiceRegistry::Operate operate(tempToken);
    //empty catalog
    edm::CatalogAttributes tmp_cat;
    //use the first catalog provided by site local config
    edm::FileLocator fl(tmp_cat, 0);
    CHECK("root://cmsxrootd.fnal.gov/store/group/bha/bho" == fl.pfn("/store/group/bha/bho"));
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
    CHECK("root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/group/bha/bho" == fl.pfn("/store/group/bha/bho"));
    for (auto file : lfn) {
      CHECK("" == fl.pfn(file));
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
    CHECK("root://host.domain//pnfs/cms/store/group/bha/bho" == fl.pfn("/store/group/bha/bho"));
    //two levels chain between "root", "second" and "first" (see storage.json)
    CHECK("root://host.domain//pnfs/cms/store/user/AAA/bho" == fl.pfn("/store/user/aaa/bho"));
    for (auto file : lfn) {
      CHECK("" == fl.pfn(file));
    }
  }
}
