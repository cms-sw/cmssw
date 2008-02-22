#ifndef FWCore_Services_SiteLocalConfigService_H
#define FWCore_Services_SiteLocalConfigService_H

//<<<<<< INCLUDES                                                       >>>>>>
#include <string>
#include <list>
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
	    const std::string calibCatalog (void) const;
	    const std::string calibLogicalServer (void) const;
	    const std::string lookupCalibConnect (const std::string& input) const;
            const std::string rfioType (void) const;

	    // implicit copy constructor
	    // implicit assignment operator
	    // implicit destructor
	private:
	    void parse (const std::string &url);
	    const std::string frontierConnect (const std::string &servlet) const;
	    std::string 	m_url;	    
	    std::string 	m_dataCatalog;
	    mutable std::string m_calibCatalog;
	    std::string		m_frontierConnect;
            std::string         m_rfioType;
	    bool		m_connected;	    
	};
    }
}

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // FRAMEWORK_SITE_LOCAL_CONFIG_H
