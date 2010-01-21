#ifndef EVF_SERVICE_WEB_REGISTRY_H
#define EVF_SERVICE_WEB_REGISTRY_H

#include <string>
#include <map>
#include <vector>
#include <iostream>
#include "xgi/Method.h"
#include "xdata/InfoSpace.h"

namespace edm{
  class ParameterSet;
}

namespace xgi{
  class Input;
  class Output;
}

namespace xdaq{
  class Application;
}

namespace evf
{
  class ServiceWeb;
  class ServiceWebRegistry
    {
    public:
      ServiceWebRegistry(const edm::ParameterSet &);

      void registerWeb(std::string &name, ServiceWeb *cl)
	{
	  std::cout << "**********registering " << name << " to service web registry " 
		    << std::hex << (unsigned long) this << std::dec << std::endl;
	  clm_.insert(std::pair<std::string, ServiceWeb*>(name,cl));
	  std::cout << "**********registry size now " << clm_.size() << std::endl;
	  // CAN ONLY BIND TO xdaq::Application methods...
	  //	  xgi::bind(cl, func, name);  
	} 
      void invoke(xgi::Input *, xgi::Output *, const std::string &);
      void publish(xdata::InfoSpace *);
      std::vector<ServiceWeb *> getWebs();

    private:
      typedef std::map<std::string, ServiceWeb*> dct;
      typedef dct::iterator idct;
      void clear();
      dct clm_;
      friend class FWEPWrapper;
    };
}
#endif

