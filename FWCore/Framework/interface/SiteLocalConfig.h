#ifndef FRAMEWORK_SITE_LOCAL_CONFIG_H
# define FRAMEWORK_SITE_LOCAL_CONFIG_H

//<<<<<< INCLUDES                                                       >>>>>>
# include <string>
# include <list>

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
  class SiteLocalConfig
  {
  public:
    typedef std::list<std::string> FrontierProxies;	    
    SiteLocalConfig () {}
    virtual ~SiteLocalConfig() {}
    
    virtual const std::string dataCatalog (void) const =0;
    virtual const std::string calibCatalog (void) const =0;
    virtual FrontierProxies::const_iterator frontierProxyBegin (void) const =0;
    virtual FrontierProxies::const_iterator frontierProxyEnd (void) const =0;
    
    // implicit copy constructor
    // implicit assignment operator
  private:
  };
}
    
//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // FRAMEWORK_SITE_LOCAL_CONFIG_H
