#ifndef CondCore_DBCommon_Connection_H
#define CondCore_DBCommon_Connection_H
#include <vector>
#include <string>
namespace coral{
  class IConnectionService;
}
namespace cond{
  class DBSession;
  class ITransaction;
  class PoolConnectionProxy;
  class CoralConnectionProxy;
  class CoralTransaction;
  class PoolTransaction;
  /*logical connection interface,each logical connection is a connection pool
    delegate real connection to underlying proxy
    developer level interface
  **/
  class Connection{
  public:
    Connection(const std::string& con,
	       const std::string& catalog,
	       unsigned int connectionTimeout=0
	       );
    Connection(const std::string& con,
	       unsigned int connectionTimeout=0);
    ~Connection();
    /// just pass  on the connection service handle to the proxy, 
    /// do not connect for real
    void connect( cond::DBSession* session );
    /// return handle to the underlying coral transaction
    CoralTransaction& coralTransaction(bool isReadOnly=true);
    /// return handle to the underlying pool transaction
    PoolTransaction& poolTransaction(bool isReadOnly=true);
    /// return connection string in use
    std::string connectStr() const;
    /// return catalog connect string in use
    std::string catalogStr() const;
  private:
    std::string m_con;
    std::string m_catalog;
    unsigned int m_connectionTimeOut;
    std::vector<PoolConnectionProxy*> m_poolConnectionPool;
    std::vector<CoralConnectionProxy*> m_coralConnectionPool;
    coral::IConnectionService* m_connectionServiceHandle;
  };// class Connection
}//ns cond
#endif
