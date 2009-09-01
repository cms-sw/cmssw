#ifndef CondCore_DBCommon_DbTransaction_H
#define CondCore_DBCommon_DbTransaction_H
//
// Package:     DBCommon
// Class  :     DbTransaction
// 
/**\class DbTransaction DbTransaction.h CondCore/DBCommon/interface/DbTransaction.h
   Description: 
*/
//
//
namespace coral 
{
  class ISessionProxy;
}

namespace pool 
{
  class ISession;
}

namespace cond{
  class DbTransaction {
  public:
    DbTransaction(coral::ISessionProxy& coralSession, pool::ISession& poolSession);   
    
    ~DbTransaction();
    /// start transaction
    int start(bool readOnly = false);
    /// commit transaction.  
    int commit();
    /// force the commit, regardless to the transaction clients
    bool forceCommit();
    /// rollback transaction
    bool rollback();
    /// current transaction is active
    int isActive() const;
    /// current transaction is readonly
    bool isReadOnly() const;
  private:
    coral::ISessionProxy& m_coralSession;
    pool::ISession& m_poolSession;
    bool m_readOnly;
    int m_clients;

  };
}
#endif
