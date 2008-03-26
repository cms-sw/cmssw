#ifndef EVF_MODULE_WEB_REGISTRY_H
#define EVF_MODULE_WEB_REGISTRY_H

#include <string>
#include <map>
#include <list>
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
  class ModuleWeb;
  class ModuleWebRegistry
    {
    public:
      ModuleWebRegistry(const edm::ParameterSet &);

      void registerWeb(std::string &name, ModuleWeb *cl)
	{
	  clm_.insert(std::pair<std::string, ModuleWeb*>(name,cl));
	  // CAN ONLY BIND TO xdaq::Application methods...
	  //	  xgi::bind(cl, func, name);  
	} 
      void invoke(xgi::Input *, xgi::Output *, const std::string &);
      void publish(xdata::InfoSpace *);
      bool checkWeb(const std::string &);

    private:
      typedef std::map<std::string, ModuleWeb*> dct;
      typedef dct::iterator idct;

      void clear();
      dct clm_;
      friend class FUEventProcessor;
    };
}
#endif

