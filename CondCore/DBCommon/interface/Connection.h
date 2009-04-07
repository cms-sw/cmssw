#ifndef CondCore_DBCommon_Connection_H
#define CondCore_DBCommon_Connection_H
//
// Package:     DBCommon
// Class  :     Connection
// 
/**\class Connection Connection.h CondCore/DBCommon/interface/Connection.h
   Description: logical connection interface. Logical connection delegates the 
   real connection to underlying proxy
*/
//
// Author:      Zhen Xie
//

#include <string>
#include <memory>

namespace pool{
  class IBlobStreamingService;
}
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
  class Connection{
  public:
    /// Constructor. If connectionTimeout=0, the commit method will close the database connection; if connectionTimeout=-1, the connection is not closed untill explict user call of disconnect method; if connectionTimeout=n, the connection will be closed by the commit method that holds the closest value of n. 
    Connection(const std::string& con,
	       int connectionTimeout=0);
    /// Destructor
    ~Connection();
    /// pass on the connection service handle to the proxy, 
    /// do not connect for real. This method must be called after 
    // DBSession::open
    void connect( cond::DBSession* session );
    /// disconnect open connection by hand if connectionTimeout <0, otherwise, no real action taken
    void disconnect();
    /// return handle to the underlying coral transaction
    CoralTransaction& coralTransaction();
    /// return handle to the underlying pool transaction
    PoolTransaction& poolTransaction();
    /// return connection string in use
    std::string connectStr() const;
  private:
    std::string m_con;
    int m_connectionTimeOut;
    int m_idleConnectionCleanupPeriod;
    std::auto_ptr<PoolConnectionProxy> m_poolConnection;
    std::auto_ptr<CoralConnectionProxy> m_coralConnection;
    coral::IConnectionService* m_connectionServiceHandle;
    pool::IBlobStreamingService* m_blobstreamingServiceHandle;
  };// class Connection
}//ns cond
#endif
