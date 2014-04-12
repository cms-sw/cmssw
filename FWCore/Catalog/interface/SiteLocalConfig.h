#ifndef FWCore_Catalog_SiteLocalConfig_h
#define FWCore_Catalog_SiteLocalConfig_h

// INCLUDES
# include <string>
# include <vector>
# include <netdb.h>

// PUBLIC DEFINES
// PUBLIC CONSTANTS
// PUBLIC TYPES
namespace edm {
    class ParameterSet;
    class ActivityRegistry;
}

// PUBLIC VARIABLES
// PUBLIC FUNCTIONS
// CLASS DECLARATIONS

namespace edm {
  class SiteLocalConfig {
  public:
    SiteLocalConfig () {}
    virtual ~SiteLocalConfig() {}

    virtual std::string const dataCatalog (void) const = 0;
    virtual std::string const fallbackDataCatalog (void) const = 0;
    virtual std::string const lookupCalibConnect (std::string const& input) const = 0;
    virtual std::string const rfioType (void) const = 0;

    virtual std::string const* sourceCacheTempDir() const = 0;
    virtual double const* sourceCacheMinFree() const = 0;
    virtual std::string const* sourceCacheHint() const = 0;
    virtual std::string const* sourceReadHint() const = 0;
    virtual unsigned int const* sourceTTreeCacheSize() const = 0;
    virtual unsigned int const* sourceTimeout() const = 0;
    virtual bool enablePrefetching() const = 0;
    virtual unsigned int debugLevel() const = 0;
    virtual std::vector<std::string> const* sourceNativeProtocols() const = 0;
    virtual struct addrinfo const * statisticsDestination() const = 0;
    virtual std::string const& siteName (void) const = 0;

    // implicit copy constructor
    // implicit assignment operator
  private:
  };
}

// INLINE PUBLIC FUNCTIONS
// INLINE MEMBER FUNCTIONS

#endif //FWCore_Catalog_SiteLocalConfig_h
