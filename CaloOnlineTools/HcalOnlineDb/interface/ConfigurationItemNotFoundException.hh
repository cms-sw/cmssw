#ifndef hcal_ConfigurationItemNotFoundException_hh_included
#define hcal_ConfigurationItemNotFoundException_hh_included 1

#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseException.hh"

namespace hcal {
  namespace exception {

    class ConfigurationItemNotFoundException: public ConfigurationDatabaseException     {
    public: 
      ConfigurationItemNotFoundException( const std::string& name, const std::string& message, const std::string& module, int line, const std::string& function ): 
	ConfigurationDatabaseException(name, message, module, line, function) 
      {} 
      
#ifdef HAVE_XDAQ
      ConfigurationItemNotFoundException( const std::string& name, const std::string& message, const std::string& module, int line, const std::string& function,
			 xcept::Exception& e ): 
	ConfigurationDatabaseException(name, message, module, line, function, e) 
      {} 
#endif
    }; 
    
  }
}

#endif // hcal_ConfigurationItemNotFoundException_hh_included
