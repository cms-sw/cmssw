#ifndef FWCore_Catalog_test_TestSiteLocalConfig_h
#define FWCore_Catalog_test_TestSiteLocalConfig_h

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

namespace edmtest::catalog {

  inline std::filesystem::path const storageDescriptionPathImpl(const edm::CatalogAttributes& aDataCatalog) {
    std::string CMSSW_BASE(std::getenv("CMSSW_BASE"));
    std::string CMSSW_RELEASE_BASE(std::getenv("CMSSW_RELEASE_BASE"));
    std::string file_name("/src/FWCore/Catalog/test/storage.json");
    std::string full_file_name = std::filesystem::exists((CMSSW_BASE + file_name).c_str())
                                     ? CMSSW_BASE + file_name
                                     : CMSSW_RELEASE_BASE + file_name;
    return full_file_name;
  }

  class TestSiteLocalConfig : public edm::SiteLocalConfig {
  public:
    //constructor using Rucio data catalogs
    TestSiteLocalConfig(std::vector<edm::CatalogAttributes> catalogs) : m_catalogs(std::move(catalogs)) {}
    std::vector<edm::CatalogAttributes> const& dataCatalogs() const final { return m_catalogs; }
    std::filesystem::path const storageDescriptionPath(const edm::CatalogAttributes& aDataCatalog) const final {
      return edmtest::catalog::storageDescriptionPathImpl(aDataCatalog);
    }

    std::string const lookupCalibConnect(std::string const& input) const final { return std::string(); }

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
    std::vector<edm::CatalogAttributes> m_catalogs;
    std::filesystem::path m_storageDescription_path;
    std::string m_emptyString;
  };

  inline std::vector<edm::CatalogAttributes> makeVectorOfCatalogAttributes() {
    std::vector<edm::CatalogAttributes> tmp;

    //catalog for testing "prefix"
    edm::CatalogAttributes aCatalog;
    aCatalog.site = "T1_US_FNAL";
    aCatalog.subSite = "T1_US_FNAL";
    aCatalog.storageSite = "T1_US_FNAL";
    aCatalog.volume = "American_Federation";
    aCatalog.protocol = "XRootD";
    tmp.push_back(aCatalog);

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

    return tmp;
  }

  inline edm::ServiceToken makeTestSiteLocalConfigToken() {
    //create the services
    return edm::ServiceToken(edm::ServiceRegistry::createContaining(
        std::unique_ptr<edm::SiteLocalConfig>(std::make_unique<TestSiteLocalConfig>(makeVectorOfCatalogAttributes()))));
  }
}  // namespace edmtest::catalog

#endif
