#ifndef FWCore_Catalog_SiteLocalConfig_h
#define FWCore_Catalog_SiteLocalConfig_h

// INCLUDES
# include <string>

// PUBLIC DEFINES
// PUBLIC CONSTANTS
// PUBLIC TYPES
namespace edm 
{
    class ParameterSet;
    class ActivityRegistry;
}

// PUBLIC VARIABLES
// PUBLIC FUNCTIONS
// CLASS DECLARATIONS

namespace edm 
{
  class SiteLocalConfig
  {
  public:
    SiteLocalConfig () {}
    virtual ~SiteLocalConfig() {}
    
    virtual const std::string dataCatalog (void) const = 0;
    virtual const std::string lookupCalibConnect (const std::string& input) const = 0;
    
    // implicit copy constructor
    // implicit assignment operator
  private:
  };
}
    
// INLINE PUBLIC FUNCTIONS
// INLINE MEMBER FUNCTIONS

#endif //FWCore_Catalog_SiteLocalConfig_h
