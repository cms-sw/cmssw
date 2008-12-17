#ifndef CondCore_DBCommon_PoolConnectionProxy_H
#define CondCore_DBCommon_PoolConnectionProxy_H
#include "CondCore/DBCommon/interface/IConnectionProxy.h"
#include "ITransactionObserver.h"
#include <string>
//
// Package:     DBCommon
// Class  :     PoolConnectionProxy
// 
/**\class PoolConnectionProxy PoolConnectionProxy.h CondCore/DBCommon/src/PoolConnectionProxy.h
   Description: this class handles pool connection. It is a IConnectionProxy implementation and a transaction observer 
*/
//
// Author:      Zhen Xie
//
namespace pool{
  class IDataSvc;
  class IFileCatalog;
  class IBlobStreamingService;
}
namespace cond{
  class ITransaction;
  class PoolTransaction;
  class PoolConnectionProxy : public IConnectionProxy, 
    public ITransactionObserver{
    public:
    PoolConnectionProxy(coral::IConnectionService* connectionServiceHandle,
			pool::IBlobStreamingService* blobStreamingServiceHandle,
			const std::string& con,
			int connectionTimeOut,
			int idleConnectionCleanupPeriod);
    ~PoolConnectionProxy();
    /// required implementation by IConnectionProxy interface
    ITransaction&  transaction();
    //bool isActive() const;
    //bool isReadOnly() const;
    //int connectionTimeOut() const;
    //std::string connectStr() const;
    pool::IDataSvc* poolDataSvc();
    /// required implementation by observer interface
    void reactOnStartOfTransaction( const ITransaction* );
    void reactOnEndOfTransaction( const ITransaction* );    
    private:
    void connect();
    void disconnect();
    private:
    pool::IBlobStreamingService* m_blobstreamingService;
    pool::IDataSvc* m_datasvc;
    cond::ITransaction* m_transaction;
    //std::string m_con;
    unsigned int m_transactionCounter;
    //int m_connectionTimeOut;
    //boost::timer m_timer;
    pool::IFileCatalog* m_catalog;
  };
}
#endif
