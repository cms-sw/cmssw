#ifndef COND_SERVICELOADER_H
#define COND_SERVICELOADER_H
#include "SealKernel/Context.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include <string>
//#include <map>
namespace seal{
  class ComponentLoader;
}
namespace  cond{
  class ConnectionConfiguration;
  //
  //loads LCG services, holds the global context for services
  //
  //
  class ServiceLoader{
  public:
    ServiceLoader();
    ~ServiceLoader();
    void usePOOLContext();
    void useOwnContext();
    seal::Context* context();
    void loadMessageService( cond::MessageLevel messagelevel=cond::Error );
    void loadAuthenticationService( cond::AuthenticationMethod method=cond::Env );
    void loadRelationalService();
    void loadConnectionService(cond::ConnectionConfiguration& config);
    /// load external streaming service
    void loadBlobStreamingService( const std::string& componentName );
    //void loadLookupService();    
    //void loadUserMonitoringService();
  private:
    void initLoader();
  private:
    bool m_isPoolContext;
    seal::Handle< seal::Context > m_context;
    seal::Handle<seal::ComponentLoader> m_loader;
  };
}//ns cond
#endif
