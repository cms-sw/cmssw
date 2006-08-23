#ifndef EVF_MODULEWEB_H
#define EVF_MODULEWEB_H

#include "toolbox/include/toolbox/lang/Class.h"
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
    private:
      std::string moduleName_;
    };
}
#endif
