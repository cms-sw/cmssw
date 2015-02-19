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

            std::string const dataCatalog(void) const override;
            std::string const fallbackDataCatalog(void) const override;
            std::string const lookupCalibConnect(std::string const& input) const override;
            std::string const rfioType(void) const override;

            std::string const* sourceCacheTempDir() const override;
            double const* sourceCacheMinFree() const override;
            std::string const* sourceCacheHint() const override;
            std::string const* sourceCloneCacheHint() const override;
            std::string const* sourceReadHint() const override;
            unsigned int const* sourceTTreeCacheSize() const override;
            unsigned int const* sourceTimeout() const override;
            bool                enablePrefetching() const override;
            unsigned int        debugLevel() const override;
            std::vector<std::string> const* sourceNativeProtocols() const override;
            struct addrinfo const* statisticsDestination() const override;
            std::set<std::string> const* statisticsInfo() const override;
            std::string const&  siteName() const override;

            // implicit copy constructor
            // implicit assignment operator
            ~SiteLocalConfigService();

            static void fillDescriptions(ConfigurationDescriptions& descriptions);

        private:
            void parse (std::string const& url);
            void computeStatisticsDestination();
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
            std::string         m_cloneCacheHint;
            std::string const*  m_cloneCacheHintPtr;
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
            std::string         m_statisticsDestination;
            struct addrinfo   * m_statisticsAddrInfo;
            static const std::string m_statisticsDefaultPort;
            std::set<std::string> m_statisticsInfo;
            bool m_statisticsInfoAvail;
            std::string         m_siteName;
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
