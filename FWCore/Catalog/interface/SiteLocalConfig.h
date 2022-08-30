#ifndef FWCore_Catalog_SiteLocalConfig_h
#define FWCore_Catalog_SiteLocalConfig_h
////////////////////////////////////////////////////////////
//
// Abstract class. dataCatalogs() returns multiple data catalogs from site-local-config.xml. It is overridden in derived classes.
//
////////////////////////////////////////////////////////////

// INCLUDES
#include <set>
#include <string>
#include <vector>
#include <netdb.h>
#include <filesystem>
// PUBLIC DEFINES
// PUBLIC CONSTANTS
// PUBLIC TYPES
namespace edm {
  class ParameterSet;
  class ActivityRegistry;

  //attributes of a data catalog (Rucio format) defined in <data-access> block of site-local-config.xml. See further description in SiteLocalConfigService.cc
  struct CatalogAttributes {
    CatalogAttributes() = default;
    CatalogAttributes(std::string input_site,
                      std::string input_subSite,
                      std::string input_storageSite,
                      std::string input_volume,
                      std::string input_protocol)
        : site(std::move(input_site)),
          subSite(std::move(input_subSite)),
          storageSite(std::move(input_storageSite)),
          volume(std::move(input_volume)),
          protocol(std::move(input_protocol)) {}
    bool operator==(const CatalogAttributes& aCatalog) const {
      return site == aCatalog.site && subSite == aCatalog.subSite && storageSite == aCatalog.storageSite &&
             volume == aCatalog.volume && protocol == aCatalog.protocol;
    }
    bool empty() const { return site.empty() && storageSite.empty() && volume.empty() && protocol.empty(); }
    std::string site;
    std::string subSite;
    std::string storageSite;  //site where storage description is used
    std::string volume;
    std::string protocol;
  };

  enum class CatalogType { TrivialCatalog, RucioCatalog };
}  // namespace edm

// PUBLIC VARIABLES
// PUBLIC FUNCTIONS
// CLASS DECLARATIONS

namespace edm {
  class SiteLocalConfig {
  public:
    SiteLocalConfig() {}
    virtual ~SiteLocalConfig() {}

    virtual std::vector<std::string> const& trivialDataCatalogs() const = 0;
    virtual std::vector<edm::CatalogAttributes> const& dataCatalogs() const = 0;
    virtual std::filesystem::path const storageDescriptionPath(const edm::CatalogAttributes& aDataCatalog) const = 0;
    virtual std::string const lookupCalibConnect(std::string const& input) const = 0;
    virtual std::string const rfioType(void) const = 0;

    virtual std::string const* sourceCacheTempDir() const = 0;
    virtual double const* sourceCacheMinFree() const = 0;
    virtual std::string const* sourceCacheHint() const = 0;
    virtual std::string const* sourceCloneCacheHint() const = 0;
    virtual std::string const* sourceReadHint() const = 0;
    virtual unsigned int const* sourceTTreeCacheSize() const = 0;
    virtual unsigned int const* sourceTimeout() const = 0;
    virtual bool enablePrefetching() const = 0;
    virtual unsigned int debugLevel() const = 0;
    virtual std::vector<std::string> const* sourceNativeProtocols() const = 0;
    virtual struct addrinfo const* statisticsDestination() const = 0;
    virtual std::set<std::string> const* statisticsInfo() const = 0;
    virtual std::string const& siteName(void) const = 0;
    virtual std::string const& subSiteName(void) const = 0;
    virtual bool useLocalConnectString() const = 0;
    virtual std::string const& localConnectPrefix() const = 0;
    virtual std::string const& localConnectSuffix() const = 0;

    // implicit copy constructor
    // implicit assignment operator
  private:
  };
}  // namespace edm

// INLINE PUBLIC FUNCTIONS
// INLINE MEMBER FUNCTIONS

#endif  //FWCore_Catalog_SiteLocalConfig_h
