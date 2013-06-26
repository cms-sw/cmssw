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
  namespace moduleweb {
    class ForkInfoObj;
  }
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
      void publishToXmas(xdata::InfoSpace *);
      bool checkWeb(const std::string &);
      void publishForkInfo(std::string name, moduleweb::ForkInfoObj *forkInfoObj);

    private:
      typedef std::map<std::string, ModuleWeb*> dct;
      typedef dct::iterator idct;
      void openBackDoor(const std::string &, unsigned int timeout_sec = 0, bool * started = 0);
      void closeBackDoor(const std::string &);
      void clear();
      dct clm_;
      friend class FWEPWrapper;
    };
}
#endif

