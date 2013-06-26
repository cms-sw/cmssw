#ifndef CondCore_DBCommon_DbOpenTransaction_H
#define CondCore_DBCommon_DbOpenTransaction_H
//
// Package:     DBCommon
// Class  :     DbOpenTransaction
// 
/**\class DbScopedTransaction DbScopedTransaction.h CondCore/DBCommon/interface/DbScopedTransaction.h
   Description: 
*/
//
//

namespace cond{

  class DbTransaction;
  
  class DbOpenTransaction {
  public:
    explicit DbOpenTransaction( cond::DbTransaction& transaction );   
    
    ~DbOpenTransaction();
    /// start transaction
    /// current transaction is readonly
    void ok();
  private:
    cond::DbTransaction& m_transaction;
    bool m_status;
    
  };
}
#endif
