#ifndef hcal_ConfigurationDatabaseException_hh_included
#define hcal_ConfigurationDatabaseException_hh_included 1

#include "CaloOnlineTools/HcalOnlineDb/interface/Exception.hh"

namespace hcal {
  namespace exception {

    class ConfigurationDatabaseException: public Exception     {
    public: 
      ConfigurationDatabaseException( const std::string& name, const std::string& message, const std::string& module, int line, const std::string& function ): 
	Exception(name, message, module, line, function) 
      {} 
      
      ConfigurationDatabaseException( const std::string& name, const std::string& message, const std::string& module, int line, const std::string& function,
			 xcept::Exception& e ): 
	Exception(name, message, module, line, function, e) 
      {} 
    }; 
    
  }
}

#endif // hcal_ConfigurationDatabaseException_hh_included
