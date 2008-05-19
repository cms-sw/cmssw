#ifndef CondCore_DBCommon_PoolTransaction_H
#define CondCore_DBCommon_PoolTransaction_H
#include "CondCore/DBCommon/interface/ITransaction.h"
//
// Package:     DBCommon
// Class  :     PoolTransaction
// 
/**\class PoolTransaction PoolTransaction.h CondCore/DBCommon/interface/PoolTransaction.h
   Description: pool transaction. PoolTransaction object is child of connection and subject for observers
*/
//
// Author:      Zhen Xie
//
namespace pool{
  class IDataSvc;
}
namespace cond{
  class PoolConnectionProxy;
  class PoolTransaction : public ITransaction{
  public:
    explicit PoolTransaction(cond::PoolConnectionProxy* parentConnection);
    ~PoolTransaction();
    /// start transaction
    void start(bool readOnly);
    /// commit transaction. Will disconnect from database if connection timeout==0 or connectted time close to the threshold  
    void commit();
    // rollback transaction
    void rollback();
    /// current transaction is active
    //virtual bool isActive() const;
    /// current transaction is readonly
    virtual bool isReadOnly() const;
    /// get handle to the parent connection
    virtual IConnectionProxy& parentConnection();
    void resetPoolDataSvc(pool::IDataSvc* datasvc) const;
    /// get pool dataSvc handle
    pool::IDataSvc& poolDataSvc();

  private:
    void upgrade();
    void forceCommit();
  private:
    cond::PoolConnectionProxy* m_parentConnection;
    mutable pool::IDataSvc* m_datasvc;
    //mutable bool m_isReadOnly;
    
    int m_count;

  };
}
#endif
