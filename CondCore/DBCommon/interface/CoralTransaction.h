#ifndef CondCore_DBCommon_CoralTransaction_H
#define CondCore_DBCommon_CoralTransaction_H
#include "CondCore/DBCommon/interface/ITransaction.h"
#include <vector>
//
// Package:     DBCommon
// Class  :     CoralTransaction
// 
/**\class CoralTransaction CoralTransaction.h CondCore/DBCommon/interface/CoralTransaction.h
   Description: coral transaction. CoralTransaction object is child of connection and subject for observers
*/
//
// Author:      Zhen Xie
//
namespace coral{
  class ISessionProxy;
  class ISchema;
}
namespace cond{
  class CoralConnectionProxy;
  class CoralTransaction : public ITransaction{
  public:
    explicit CoralTransaction(CoralConnectionProxy* parentConnection);
    ~CoralTransaction();
    /// start transaction
    virtual void start(bool isReadOnly);
    /// commit transaction. Will disconnect from database if connection timeout==0 or connectted time close to the threshold  
    virtual void commit();
    /// rollback transaction
    virtual void rollback();
    /// current transaction is active
    //virtual bool isActive() const;
    /// current transaction is readonly
    virtual bool isReadOnly() const;
    /// get handle to the parent connection
    virtual IConnectionProxy& parentConnection();
    /// get nominal schema for the transaction
    coral::ISchema& nominalSchema();
    /// get coralSessionProxy handle 
    coral::ISessionProxy& coralSessionProxy();
    void resetCoralHandle(coral::ISessionProxy* coralHandle) const;
  private:
    CoralConnectionProxy* m_parentConnection;
    /// coral sessionproxy handle. parent connection has the ownership
    mutable coral::ISessionProxy* m_coralHandle;
    mutable bool m_isReadOnly;
  };
}
#endif
