#ifndef CondCore_DBCommon_CoralConnectionProxy_H
#define CondCore_DBCommon_CoralConnectionProxy_H
#include "CondCore/DBCommon/interface/IConnectionProxy.h"
#include "ITransactionObserver.h"
#include <string>
//
// Package:     DBCommon
// Class  :     CoralConnectionProxy
// 
/**\class CoralConnectionProxy CoralConnectionProxy.h CondCore/DBCommon/src/CoralConnectionProxy.h
   Description: this class handles coral connection. It is a IConnectionProxy implementation and a transaction observer 
*/
//
// Author:      Zhen Xie
//
namespace coral{
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
			 int connectionTimeOut
			 );
    ~CoralConnectionProxy();
    /// required implementation by IConnectionProxy interface
    ITransaction&  transaction();
    /// returns coral session proxy
    coral::ISessionProxy& coralProxy();
    /// required implementation by observer interface
    void reactOnStartOfTransaction( const ITransaction* );
    void reactOnEndOfTransaction( const ITransaction* );    
    private:
    //coral::IConnectionService* m_connectionSvcHandle;
    //std::string m_con;
    coral::ISessionProxy* m_coralHandle;
    unsigned int m_transactionCounter;
    //int m_connectionTimeOut;
    cond::CoralTransaction* m_transaction;
    //boost::timer m_timer;
  private:
    void connect(bool isReadOnly);
    void disconnect();
  };
}
#endif
