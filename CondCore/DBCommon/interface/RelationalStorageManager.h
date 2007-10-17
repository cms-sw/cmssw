#ifndef COND_DBCommon_RelationalStorageManager_h
#define COND_DBCommon_RelationalStorageManager_h
#include <string>
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "RelationalAccess/IConnectionService.h"
#include "SealKernel/Context.h"
namespace coral{
  class ISessionProxy;
}
namespace cond{
  class DBSession;
  /**
   * Class manages pure CORAL session and transaction 
   */
  class RelationalStorageManager{
  public:
    explicit RelationalStorageManager(const std::string& con);
    RelationalStorageManager(const std::string& con,
			     cond::DBSession* session);
    ~RelationalStorageManager();
    DBSession& session();
    bool isSessionShared() const;
    coral::ISessionProxy* connect(cond::ConnectMode mod);
    void disconnect();
    void startTransaction(bool isReadOnly=true);
    void commit();
    void rollback();
    std::string connectionString() const;
    coral::ISessionProxy& sessionProxy();
  private:
    std::string m_con;
    coral::ISessionProxy* m_proxy;
    bool m_readOnlyMode;
    bool m_started;
    DBSession* m_sessionHandle;
    bool m_sessionShared;
    //seal::IHandle<coral::IConnectionService> m_connectionService;
  private:
    void init();
    seal::IHandle<coral::IConnectionService> connectionService();
  };
}//ns cond
#endif
