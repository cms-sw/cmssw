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
namespace edm 
{
    class ParameterSet;
    class ActivityRegistry;
}

//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

namespace edm 
{
    namespace service 
    {
      class SiteLocalConfigService : public SiteLocalConfig
	{
	public:
	    SiteLocalConfigService (const ParameterSet & pset,
			     const ActivityRegistry &activityRegistry);
	    
	    const std::string dataCatalog (void) const;
	    const std::string lookupCalibConnect (const std::string& input) const;
	    const std::string rfioType (void) const;

            const std::string* sourceCacheTempDir() const;
            const double* sourceCacheMinFree() const;
            const std::string* sourceCacheHint() const;
            const std::string* sourceReadHint() const;
            const unsigned int* sourceTTreeCacheSize() const;
            const std::vector<std::string>* sourceNativeProtocols() const;
	    // implicit copy constructor
	    // implicit assignment operator
	    // implicit destructor
	private:
	    void parse (const std::string &url);
	    const std::string frontierConnect (const std::string &servlet) const;
	    std::string 	m_url;	    
	    std::string 	m_dataCatalog;
	    std::string		m_frontierConnect;
	    std::string 	m_rfioType;
	    bool		m_connected;	    
            std::string         m_cacheTempDir;
            std::string const*  m_cacheTempDirPtr;
	    double		m_cacheMinFree;
	    double const *	m_cacheMinFreePtr;
            std::string         m_cacheHint;
            std::string const*  m_cacheHintPtr;
            std::string         m_readHint;
            std::string const*  m_readHintPtr;
            unsigned int        m_ttreeCacheSize;
            unsigned int const* m_ttreeCacheSizePtr;
            std::vector<std::string> m_nativeProtocols;
            std::vector<std::string> const* m_nativeProtocolsPtr;
 	};
    }
}

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // FRAMEWORK_SITE_LOCAL_CONFIG_H
