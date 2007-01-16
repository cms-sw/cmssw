#ifndef COND_DBCommon_PoolStorageManager_h
#define COND_DBCommon_PoolStorageManager_h
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include <string>
#include <vector>
namespace pool{
  class IFileCatalog;
  class IDataSvc;
  class IDatabase;
}
namespace cond{
  class DBSession;
  /*
   * Class manages POOL session and transaction. Holds DataSvc 
   **/
  class PoolStorageManager{
  public:
    PoolStorageManager(const std::string& con,
		       const std::string& catalog);
    PoolStorageManager(const std::string& con,
		       const std::string& catalog, 
		       DBSession* session);
    ~PoolStorageManager();
    DBSession& session();
    bool isSessionShared() const;
    void connect();
    void disconnect();
    void startTransaction(bool isReadOnly=true);
    void commit();
    void rollback();
    std::string catalogString() const;
    std::string connectionString() const;
    std::vector<std::string> containers();
    pool::IDataSvc& DataSvc();
   //ContainerIterator* newContainerIterator(const std::string& containername);
  private:
    std::string m_catalogstr;
    std::string m_con;
    pool::IFileCatalog* m_cat;
    pool::IDataSvc* m_svc;
    pool::IDatabase* m_db;
    bool m_started;
    DBSession* m_sessionHandle;
    bool m_sessionShared;
  private:
    void init();
  };
}//ns cond
#endif
