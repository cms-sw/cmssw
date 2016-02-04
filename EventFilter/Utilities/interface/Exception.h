#ifndef EVF_XDAQ_EXCEPTION_H
#define EVF_XDAQ_EXCEPTION_H

#include "xcept/Exception.h"

namespace evf
{
  class Exception : public xcept::Exception
    {
    public: 
      Exception( std::string name, std::string message, std::string module, int line, std::string function ): 
	xcept::Exception(name, message, module, 
			 line, function) 
	{} 
      
      Exception( std::string name, std::string message, std::string module, int line, std::string function,
		 xcept::Exception& e ): 
	xcept::Exception(name, message, module, 
line, function, e) 
	{} 
      
    };
}
#endif
