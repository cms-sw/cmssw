#ifndef EVF_MODULEWEB_H
#define EVF_MODULEWEB_H

#include "toolbox/lang/Class.h"
#include "xdata/InfoSpace.h"
#include <string>

namespace xgi{
  class Input;
  class Output;
}


namespace evf
{
  class ModuleWeb : public toolbox::lang::Class
    {
    public:
      ModuleWeb(const std::string &);
      virtual ~ModuleWeb(){}
      virtual void defaultWebPage(xgi::Input *in, xgi::Output *out); 
      virtual void publish(xdata::InfoSpace *) = 0;
    protected:
      std::string moduleName_;
    };
}
#endif
