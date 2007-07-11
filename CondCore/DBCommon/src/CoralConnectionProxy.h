#ifndef CondCore_DBCommon_CoralConnectionProxy_H
#define CondCore_DBCommon_CoralConnectionProxy_H
#include "CondCore/DBCommon/interface/IConnectionProxy.h"
#include "ITransactionObserver.h"
#include <string>
#include <vector>
#include <boost/timer.hpp>
namespace coral{
  class IConnectionService;
  class ISessionProxy;
}
namespace cond{
  class ITransaction;
  class CoralTransaction;
  class CoralConnectionProxy : public IConnectionProxy, 
    public ITransactionObserver{
  public:
    CoralConnectionProxy(coral::IConnectionService* connectionServiceHandle,
			 const std::string& con,
			 bool isReadOnly,
			 unsigned int connectionTimeOut
			 );
    ~CoralConnectionProxy();
    ///connection interface
    ITransaction&  transaction();
    bool isReadOnly() const ;
    unsigned int connectionTimeOut() const;
    coral::ISessionProxy& coralProxy();
    std::string connectStr() const;
    ///observer interface
    void reactOnStartOfTransaction( const ITransaction* );
    void reactOnEndOfTransaction( const ITransaction* );    
  private:
    coral::IConnectionService* m_connectionSvcHandle;
    std::string m_con;
    bool m_isReadOnly;
    coral::ISessionProxy* m_coralHandle;
    unsigned int m_transactionCounter;
    unsigned int m_connectionTimeOut;
    cond::CoralTransaction* m_transaction;
    boost::timer m_timer;
  private:
    void connect();
    void disconnect();
  };
}
#endif
