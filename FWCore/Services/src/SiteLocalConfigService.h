#ifndef FWCore_Services_SiteLocalConfigService_H
#define FWCore_Services_SiteLocalConfigService_H

//<<<<<< INCLUDES                                                       >>>>>>
#include <string>
#include <list>
#include <vector>
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include <xercesc/util/XercesDefs.hpp>
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
            edm::propagate_const<struct addrinfo*> m_statisticsAddrInfo;
            static const std::string m_statisticsDefaultPort;
            std::set<std::string> m_statisticsInfo;
            bool m_statisticsInfoAvail;
            std::string         m_siteName;
	    XMLCh *m_str_site;
	    XMLCh *m_str_name;
	    XMLCh *m_str_event_data;
	    XMLCh *m_str_catalog;
	    XMLCh *m_str_url;
	    XMLCh *m_str_rfiotype;
	    XMLCh *m_str_value;
	    XMLCh *m_str_calib_data;
	    XMLCh *m_str_frontier_connect;
	    XMLCh *m_str_source_config;
	    XMLCh *m_str_cache_temp_dir;
	    XMLCh *m_str_cache_min_free;
	    XMLCh *m_str_cache_hint;
	    XMLCh *m_str_clone_cache_hint;
	    XMLCh *m_str_read_hint;
	    XMLCh *m_str_ttree_cache_size;
	    XMLCh *m_str_timeout_in_seconds;
	    XMLCh *m_str_statistics_destination;
	    XMLCh *m_str_endpoint;
	    XMLCh *m_str_info;
	    XMLCh *m_str_prefetching;
	    XMLCh *m_str_native_protocols;
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
