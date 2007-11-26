#ifndef CondCore_DBCommon_PoolConnectionProxy_H
#define CondCore_DBCommon_PoolConnectionProxy_H
#include "CondCore/DBCommon/interface/IConnectionProxy.h"
#include "ITransactionObserver.h"
#include <string>
#include <boost/timer.hpp>
namespace pool{
  class IDataSvc;
  class IFileCatalog;
}
namespace cond{
  class ITransaction;
  class PoolTransaction;
  class PoolConnectionProxy : public IConnectionProxy, 
    public ITransactionObserver{
    public:
    PoolConnectionProxy(const std::string& con,
			int connectionTimeOut);
    ~PoolConnectionProxy();
    ///connection interface
    ITransaction&  transaction();
    //bool isActive() const;
    bool isReadOnly() const;
    int connectionTimeOut() const;
    std::string connectStr() const;
    pool::IDataSvc* poolDataSvc();
    ///observer interface
    void reactOnStartOfTransaction( const ITransaction* );
    void reactOnEndOfTransaction( const ITransaction* );    
  private:
    void connect();
    void disconnect();
  private:
    pool::IDataSvc* m_datasvc;
    cond::ITransaction* m_transaction;
    std::string m_con;
    unsigned int m_transactionCounter;
    int m_connectionTimeOut;
    boost::timer m_timer;
    pool::IFileCatalog* m_catalog;
  };
}
#endif
