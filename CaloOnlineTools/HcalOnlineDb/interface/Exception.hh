#ifndef hcal_Exception_hh_included
#define hcal_Exception_hh_included 1

#include "xcept/Exception.h"

namespace hcal {
  namespace exception {

    class Exception: public xcept::Exception {
    public: 
      Exception( const std::string& name, const std::string& message, const std::string& module, int line, const std::string& function ): 
	xcept::Exception(name, message, module, line, function) 
      {} 
      
      Exception( const std::string& name, const std::string& message, const std::string& module, int line, const std::string& function,
		 xcept::Exception& e ): 
	xcept::Exception(name, message, module, line, function, e) 
      {} 
    };     
  }
}

#endif // hcal_Exception_hh_included
