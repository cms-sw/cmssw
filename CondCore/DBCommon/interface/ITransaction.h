#ifndef CondCore_DBCommon_ITransaction_H
#define CondCore_DBCommon_ITransaction_H
#include <vector>
//
// Package:     DBCommon
// Class  :     ITransaction
// 
/**\class ITransaction ITransaction.h CondCore/DBCommon/interface/ITransaction.h
   Description: abstract transaction interface 
*/
//
// Author:      Zhen Xie
//
namespace cond{
  class IConnectionProxy;
  class ITransactionObserver;
  class ITransaction{
  public:
    ITransaction(){
      m_observers.reserve(10);
    }
    virtual ~ITransaction(){}
    /// start transaction
    virtual void start(bool isReadOnly=true) = 0;
    /// commit transaction. Will disconnect from database if connection timeout==0 or connectted time close to the threshold  
    virtual void commit() = 0;
    /// rollback transaction
    virtual void rollback() = 0;
    // current transaction is active
    //virtual bool isActive() const = 0;
    // current transaction is readonly
    virtual bool isReadOnly() const = 0;
    // get handle to the parent connection
    virtual IConnectionProxy& parentConnection() = 0;
  protected:
    void attach( ITransactionObserver* );
    void NotifyStartOfTransaction( ) ;
    void NotifyEndOfTransaction() ;
  private:
    std::vector< cond::ITransactionObserver* > m_observers;
  };
}
#endif
