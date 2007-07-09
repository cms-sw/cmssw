#ifndef COND_DBCommon_DBSession_h
#define COND_DBCommon_DBSession_h
#include <string>
#include "SealKernel/Context.h"
#include "SealKernel/ComponentLoader.h"
namespace coral{
  class IConnectionService;
  class IRelationalService;
  class IAuthenticationService;
  class IMonitoringReporter;
  class IWebCacheControl;
}
namespace pool{
  class IBlobStreamingService;
}
namespace cond{
  class SessionConfiguration;
  /*
  **/
  class DBSession{
  public:
    DBSession();
    ~DBSession();
    void open();
    //void close();
    coral::IConnectionService& connectionService();
    coral::IRelationalService& relationalService();
    coral::IAuthenticationService& authenticationService() const;
    const coral::IMonitoringReporter& monitoringReporter() const;
    coral::IWebCacheControl& webCacheControl();
    pool::IBlobStreamingService& blobStreamingService();
    cond::SessionConfiguration& configuration();
  private:
    seal::Handle<seal::Context> m_context;
    seal::Handle<seal::ComponentLoader> m_loader;
    coral::IConnectionService* m_con;
    SessionConfiguration* m_sessionConfig;
  };
}//ns cond
#endif
// DBSESSION_H
