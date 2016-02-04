#ifndef CondCore_DBCommon_DbScopedTransaction_H
#define CondCore_DBCommon_DbScopedTransaction_H
//
// Package:     DBCommon
// Class  :     DbScopedTransaction
// 
/**\class DbScopedTransaction DbScopedTransaction.h CondCore/DBCommon/interface/DbScopedTransaction.h
   Description: 
*/
//
//

namespace cond{

  class DbSession;
  
  class DbScopedTransaction {
  public:
    explicit DbScopedTransaction( cond::DbSession& session );   
    
    ~DbScopedTransaction();
    /// start transaction
    int start(bool readOnly = false);
    /// commit transaction. Will disconnect from database if connection timeout==0 or connectted time close to the threshold  
    int commit();
    /// rollback transaction
    bool rollback();
    /// query if locally has been activated
    bool isLocallyActive() const;
    /// current transaction is active
    int isActive() const;
    /// current transaction is readonly
    bool isReadOnly() const;
  private:
    cond::DbSession& m_session;
    bool m_locallyActive;
    
  };
}
#endif
