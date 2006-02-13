#ifndef COND_SERVICELOADER_H
#define COND_SERVICELOADER_H
//#include "SealKernel/Context.h"
//#include "SealKernel/ComponentLoader.h"
#include "AuthenticationMethod.h"
#include "MessageLevel.h"
#include <string>
#include <map>
namespace coral{
  class IRelationalService;
  class IAuthenticationService;
}
namespace seal{
  class Context;
  class IMessageService;
  class ComponentLoader;
  //class Handle;
  //class Component;
}
namespace cond{
  //
  //wrapper around loading LCG services
  //
  class ServiceLoader{
  public:
    /// factory methold. hand over the ownership to user;
    ServiceLoader();
    ~ServiceLoader();
    seal::IMessageService& loadMessageService( cond::MessageLevel level=cond::Error );
    coral::IAuthenticationService& loadAuthenticationService( cond::AuthenticationMethod method=cond::Env );
    coral::IRelationalService& loadRelationalService();
    void loadConnectionService();
    /// load the default streaming service
    void loadBlobStreamingService();
    /// load external streaming service
    void loadBlobStreamingService( const std::string& componentName );
  private:
    seal::Context* m_context;
    seal::ComponentLoader* m_loader;
    //Components		m_components;
  };
}//ns cond
#endif
