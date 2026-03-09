#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"

#include "TestSiteLocalConfig.h"

namespace {
  std::unique_ptr<edm::FileLocator> makeFileLocator(unsigned int iCatalog) {
    auto vec = edmtest::catalog::makeVectorOfCatalogAttributes();
    edm::CatalogAttributes const& catalogAttributes = vec.at(iCatalog);
    std::filesystem::path filename_storage = edmtest::catalog::storageDescriptionPathImpl(catalogAttributes);
    return std::make_unique<edm::FileLocator>(catalogAttributes, filename_storage);
  }
}  // namespace

TEST_CASE("FileLocator with Rucio data catalog", "[FWCore/Catalog]") {
  SECTION("prefix") {
    auto fl = makeFileLocator(0);
    CHECK("root://cmsxrootd.fnal.gov/store/group/bha/bho" == fl->pfn("/store/group/bha/bho"));
  }
  SECTION("rule") {
    auto fl = makeFileLocator(1);
    const std::array<const char*, 7> lfn = {{"/bha/bho",
                                             "bha",
                                             "file:bha",
                                             "file:/bha/bho",
                                             "/castor/cern.ch/cms/bha/bho",
                                             "someprotocol:/castor/cern.ch/cms/bha/bho",
                                             "someprotocol:/bha/bho"}};
    CHECK("root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/group/bha/bho" == fl->pfn("/store/group/bha/bho"));
    for (auto file : lfn) {
      CHECK("" == fl->pfn(file));
    }
  }
  SECTION("chainedrule") {
    auto fl = makeFileLocator(2);
    const std::array<const char*, 7> lfn = {{"/bha/bho",
                                             "bha",
                                             "file:bha",
                                             "file:/bha/bho",
                                             "/castor/cern.ch/cms/bha/bho",
                                             "someprotocol:/castor/cern.ch/cms/bha/bho",
                                             "someprotocol:/bha/bho"}};
    //one level chain between "root" and "second" protocols (see storage.json)
    CHECK("root://host.domain//pnfs/cms/store/group/bha/bho" == fl->pfn("/store/group/bha/bho"));
    //two levels chain between "root", "second" and "first" (see storage.json)
    CHECK("root://host.domain//pnfs/cms/store/user/AAA/bho" == fl->pfn("/store/user/aaa/bho"));
    for (auto file : lfn) {
      CHECK("" == fl->pfn(file));
    }
  }
}
