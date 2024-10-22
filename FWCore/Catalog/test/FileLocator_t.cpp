#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include <filesystem>

#include <string>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

namespace {
  class TestSiteLocalConfig : public edm::SiteLocalConfig {
  public:
    //constructor using trivial data catalogs
    TestSiteLocalConfig(std::vector<std::string> catalogs) : m_trivialCatalogs(std::move(catalogs)) {}
    //constructor using Rucio data catalogs
    TestSiteLocalConfig(std::vector<edm::CatalogAttributes> catalogs) : m_catalogs(std::move(catalogs)) {}
    std::vector<std::string> const& trivialDataCatalogs() const final { return m_trivialCatalogs; }
    std::vector<edm::CatalogAttributes> const& dataCatalogs() const final { return m_catalogs; }
    std::filesystem::path const storageDescriptionPath(const edm::CatalogAttributes& aDataCatalog) const final {
      return std::filesystem::path();
    }

    std::string const lookupCalibConnect(std::string const& input) const final { return std::string(); }
    std::string const rfioType(void) const final { return std::string(); }

    std::string const* sourceCacheTempDir() const final { return nullptr; }
    double const* sourceCacheMinFree() const final { return nullptr; }
    std::string const* sourceCacheHint() const final { return nullptr; }
    std::string const* sourceCloneCacheHint() const final { return nullptr; }
    std::string const* sourceReadHint() const final { return nullptr; }
    unsigned int const* sourceTTreeCacheSize() const final { return nullptr; }
    unsigned int const* sourceTimeout() const final { return nullptr; }
    bool enablePrefetching() const final { return false; }
    unsigned int debugLevel() const final { return 0; }
    std::vector<std::string> const* sourceNativeProtocols() const final { return nullptr; }
    struct addrinfo const* statisticsDestination() const final { return nullptr; }
    std::set<std::string> const* statisticsInfo() const final { return nullptr; }
    std::string const& siteName(void) const final { return m_emptyString; }
    std::string const& subSiteName(void) const final { return m_emptyString; }
    bool useLocalConnectString() const final { return false; }
    std::string const& localConnectPrefix() const final { return m_emptyString; }
    std::string const& localConnectSuffix() const final { return m_emptyString; }

  private:
    std::vector<std::string> m_trivialCatalogs;
    std::vector<edm::CatalogAttributes> m_catalogs;
    std::filesystem::path m_storageDescription_path;
    std::string m_emptyString;
  };
}  // namespace

TEST_CASE("FileLocator with Rucio data catalog", "[filelocatorRucioDataCatalog]") {
  //catalog for testing "prefix"
  edm::CatalogAttributes aCatalog;
  aCatalog.site = "T1_US_FNAL";
  aCatalog.subSite = "T1_US_FNAL";
  aCatalog.storageSite = "T1_US_FNAL";
  aCatalog.volume = "American_Federation";
  aCatalog.protocol = "XRootD";
  std::vector<edm::CatalogAttributes> tmp{aCatalog};
  //catalog for testing "rules"
  aCatalog.site = "T1_US_FNAL";
  aCatalog.subSite = "T1_US_FNAL";
  aCatalog.storageSite = "T1_US_FNAL";
  aCatalog.volume = "FNAL_dCache_EOS";
  aCatalog.protocol = "XRootD";
  tmp.push_back(aCatalog);
  //catalog for testing chained "rules"
  aCatalog.site = "T1_US_FNAL";
  aCatalog.subSite = "T1_US_FNAL";
  aCatalog.storageSite = "T1_US_FNAL";
  aCatalog.volume = "FNAL_dCache_EOS";
  aCatalog.protocol = "root";
  tmp.push_back(aCatalog);

  //create the services
  edm::ServiceToken tempToken(
      edm::ServiceRegistry::createContaining(std::unique_ptr<edm::SiteLocalConfig>(new TestSiteLocalConfig(tmp))));

  std::string CMSSW_BASE(std::getenv("CMSSW_BASE"));
  std::string CMSSW_RELEASE_BASE(std::getenv("CMSSW_RELEASE_BASE"));
  std::string file_name("/src/FWCore/Catalog/test/storage.json");
  std::string full_file_name = std::filesystem::exists((CMSSW_BASE + file_name).c_str())
                                   ? CMSSW_BASE + file_name
                                   : CMSSW_RELEASE_BASE + file_name;

  SECTION("prefix") {
    edm::ServiceRegistry::Operate operate(tempToken);
    //empty catalog
    edm::CatalogAttributes tmp_cat;
    //use the first catalog provided by site local config
    edm::FileLocator fl(tmp_cat, 0, full_file_name);
    CHECK("root://cmsxrootd.fnal.gov/store/group/bha/bho" ==
          fl.pfn("/store/group/bha/bho", edm::CatalogType::RucioCatalog));
  }
  SECTION("rule") {
    edm::ServiceRegistry::Operate operate(tempToken);
    //empty catalog
    edm::CatalogAttributes tmp_cat;
    //use the second catalog provided by site local config
    edm::FileLocator fl(tmp_cat, 1, full_file_name);
    const std::array<const char*, 7> lfn = {{"/bha/bho",
                                             "bha",
                                             "file:bha",
                                             "file:/bha/bho",
                                             "/castor/cern.ch/cms/bha/bho",
                                             "rfio:/castor/cern.ch/cms/bha/bho",
                                             "rfio:/bha/bho"}};
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
    edm::FileLocator fl(tmp_cat, 2, full_file_name);
    const std::array<const char*, 7> lfn = {{"/bha/bho",
                                             "bha",
                                             "file:bha",
                                             "file:/bha/bho",
                                             "/castor/cern.ch/cms/bha/bho",
                                             "rfio:/castor/cern.ch/cms/bha/bho",
                                             "rfio:/bha/bho"}};
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

TEST_CASE("FileLocator", "[filelocator]") {
  std::string CMSSW_BASE(std::getenv("CMSSW_BASE"));
  std::string CMSSW_RELEASE_BASE(std::getenv("CMSSW_RELEASE_BASE"));
  std::string file_name("/src/FWCore/Catalog/test/simple_catalog.xml");
  std::string full_file_name = std::filesystem::exists((CMSSW_BASE + file_name).c_str())
                                   ? CMSSW_BASE + file_name
                                   : CMSSW_RELEASE_BASE + file_name;

  //create the services
  std::vector<std::string> tmp{std::string("trivialcatalog_file:") + full_file_name + "?protocol=xrd"};
  edm::ServiceToken tempToken(
      edm::ServiceRegistry::createContaining(std::unique_ptr<edm::SiteLocalConfig>(new TestSiteLocalConfig(tmp))));

  //make the services available
  SECTION("standard") {
    edm::ServiceRegistry::Operate operate(tempToken);
    edm::FileLocator fl("");

    const std::array<const char*, 7> lfn = {{"/bha/bho",
                                             "bha",
                                             "file:bha",
                                             "file:/bha/bho",
                                             "/castor/cern.ch/cms/bha/bho",
                                             "rfio:/castor/cern.ch/cms/bha/bho",
                                             "rfio:/bha/bho"}};

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
                                       "rfio:/castor/cern.ch/cms/bha/bho",
                                       "rfio:/bha/bho"}};

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
