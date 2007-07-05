#ifndef COND_DBCommon_DBSession_h
#define COND_DBCommon_DBSession_h
#include <string>
#include "SealKernel/Context.h"
#include "SealKernel/ComponentLoader.h"
namespace coral{
  class IConnectionService;
  class IRelationalService;
  class IAuthenticationService;
}
namespace cond{
  class SessionConfiguration;
  class ConnectionConfiguration;
  /*
  **/
  class DBSession{
  public:
    DBSession();
    ~DBSession();
    void open();
    //void close();
    coral::IConnectionService& connectionServiceHandle(){return *m_con;}
    coral::IRelationalService& relationalServiceHandle();
    coral::IAuthenticationService& authenticationServiceHandle();
    //get context handle
    //SessionConfiguration& sessionConfiguration();
    //bool isActive() const;
    //void purgeConnectionPool();
  private:
    seal::Handle<seal::Context> m_context;
    seal::Handle<seal::ComponentLoader> m_loader;
    coral::IConnectionService* m_con;
    //bool m_isActive;
    ConnectionConfiguration* m_connectConfig;
    SessionConfiguration* m_sessionConfig;
    //bool m_usePoolContext;
  };
}//ns cond
#endif
// DBSESSION_H
