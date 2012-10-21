#ifndef FWCore_Services_SiteLocalConfigService_H
#define FWCore_Services_SiteLocalConfigService_H

//<<<<<< INCLUDES                                                       >>>>>>
#include <string>
#include <list>
#include <vector>
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
namespace edm {
    class ParameterSet;
}

//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

namespace edm {
    class ConfigurationDescriptions;
    namespace service {
      class SiteLocalConfigService : public SiteLocalConfig {
        public:
            explicit SiteLocalConfigService(ParameterSet const& pset);

            std::string const dataCatalog(void) const;
            std::string const fallbackDataCatalog(void) const;
            std::string const lookupCalibConnect(std::string const& input) const;
            std::string const rfioType(void) const;

            std::string const* sourceCacheTempDir() const;
            double const* sourceCacheMinFree() const;
            std::string const* sourceCacheHint() const;
            std::string const* sourceReadHint() const;
            unsigned int const* sourceTTreeCacheSize() const;
            unsigned int const* sourceTimeout() const;
            bool                enablePrefetching() const;
            unsigned int        debugLevel() const;
            std::vector<std::string> const* sourceNativeProtocols() const;
            // implicit copy constructor
            // implicit assignment operator
            // implicit destructor

            static void fillDescriptions(ConfigurationDescriptions& descriptions);

        private:
            void parse (std::string const& url);
            std::string const frontierConnect(std::string const& servlet) const;
            std::string         m_url;
            std::string         m_dataCatalog;
            std::string         m_fallbackDataCatalog;
            std::string         m_frontierConnect;
            std::string         m_rfioType;
            bool                m_connected;
            std::string         m_cacheTempDir;
            std::string const*  m_cacheTempDirPtr;
            double              m_cacheMinFree;
            double const*       m_cacheMinFreePtr;
            std::string         m_cacheHint;
            std::string const*  m_cacheHintPtr;
            std::string         m_readHint;
            std::string const*  m_readHintPtr;
            unsigned int        m_ttreeCacheSize;
            unsigned int const* m_ttreeCacheSizePtr;
            unsigned int        m_timeout;
            unsigned int const* m_timeoutPtr;
            unsigned int        m_debugLevel;
            bool                m_enablePrefetching;
            bool const        * m_enablePrefetchingPtr;
            std::vector<std::string> m_nativeProtocols;
            std::vector<std::string> const* m_nativeProtocolsPtr;
         };

         inline
         bool isProcessWideService(SiteLocalConfigService const*) {
           return true;
         }
    }
}

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // FRAMEWORK_SITE_LOCAL_CONFIG_H
