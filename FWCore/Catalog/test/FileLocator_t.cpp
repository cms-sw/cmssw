#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include <boost/filesystem.hpp>

#include <string>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

namespace {
  class TestSiteLocalConfig : public edm::SiteLocalConfig {
  public:
    TestSiteLocalConfig(std::vector<std::string> catalogs) : m_catalogs(std::move(catalogs)) {}
    std::vector<std::string> const& dataCatalogs(void) const final { return m_catalogs; }
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
    struct addrinfo const* statisticsDestination() const final {
      return nullptr;
    }
    std::set<std::string> const* statisticsInfo() const final { return nullptr; }
    std::string const& siteName(void) const final { return m_emptyString; }
    bool useLocalConnectString() const final { return false; }
    std::string const& localConnectPrefix() const final { return m_emptyString; }
    std::string const& localConnectSuffix() const final { return m_emptyString; }

  private:
    std::vector<std::string> m_catalogs;
    std::string m_emptyString;
  };
}  // namespace

TEST_CASE("FileLocator", "[filelocator]") {
  std::string CMSSW_BASE(std::getenv("CMSSW_BASE"));
  std::string CMSSW_RELEASE_BASE(std::getenv("CMSSW_RELEASE_BASE"));
  std::string file_name("/src/FWCore/Catalog/test/simple_catalog.xml");
  std::string full_file_name = boost::filesystem::exists((CMSSW_BASE + file_name).c_str())
                                   ? CMSSW_BASE + file_name
                                   : CMSSW_RELEASE_BASE + file_name;

  //create the services
  std::vector<std::string> tmp{std::string("trivialcatalog_file:") + full_file_name + "?protocol=xrd"};
  edm::ServiceToken tempToken(
      edm::ServiceRegistry::createContaining(std::unique_ptr<edm::SiteLocalConfig>(new TestSiteLocalConfig(tmp))));

  //make the services available
  SECTION("standard") {
    edm::ServiceRegistry::Operate operate(tempToken);
    edm::FileLocator fl("", 0);

    const std::array<const char*, 7> lfn = {{"/bha/bho",
                                             "bha",
                                             "file:bha",
                                             "file:/bha/bho",
                                             "/castor/cern.ch/cms/bha/bho",
                                             "rfio:/castor/cern.ch/cms/bha/bho",
                                             "rfio:/bha/bho"}};

    CHECK("/storage/path/store/group/bha/bho" == fl.pfn("/store/group/bha/bho"));
    for (auto file : lfn) {
      CHECK("" == fl.pfn(file));
    }

    CHECK(fl.lfn("/storage/path/store/group/bha/bho") == "/store/group/bha/bho");
    CHECK(fl.lfn("/store/group/bha/bho") == "");
    for (auto file : lfn) {
      CHECK("" == fl.lfn(file));
    }
  }

  SECTION("override") {
    edm::ServiceRegistry::Operate operate(tempToken);

    std::string override_file_name("/src/FWCore/Catalog/test/override_catalog.xml");
    std::string override_full_file_name = boost::filesystem::exists((CMSSW_BASE + override_file_name).c_str())
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

    CHECK("/FULL_PATH_TO_THE_FIRST_STEP_ROOT_FILE/80EC0BCD-D279-DF11-B1DB-0030487C90EE.root" == fl.pfn(overriden_file));

    for (auto f : lfn) {
      CHECK("" == fl.pfn(f));
    }
    for (auto f : lfn) {
      CHECK("" == fl.lfn(f));
    }
  }
}
