#ifndef EVF_SERVICEWEB_H
#define EVF_SERVICEWEB_H

#include "toolbox/lang/Class.h"
#include "xdata/InfoSpace.h"
#include <string>

namespace xgi{
  class Input;
  class Output;
}


namespace evf
{
  class ServiceWeb : public toolbox::lang::Class
    {
    public:
      ServiceWeb(const std::string &);
      virtual ~ServiceWeb(){}
      virtual void defaultWebPage(xgi::Input *in, xgi::Output *out); 
      virtual void publish(xdata::InfoSpace *) = 0;
      virtual void publishToXmas(xdata::InfoSpace *){};
      std::string const &name()const { return serviceName_;}
    protected:
      std::string serviceName_;
    private:
      friend class ServiceWebRegistry;
    };
}
#endif
