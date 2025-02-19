#ifndef INCLUDE_ORA_SCOPEDTRANSACTION_H
#define INCLUDE_ORA_SCOPEDTRANSACTION_H

namespace ora {

  class Transaction;
  
  /** @class ScopedTransaction ScopedTransaction.h 
   *
   */

  class ScopedTransaction {

    public:
    explicit ScopedTransaction( Transaction& dbTransaction );

    ///
    ScopedTransaction( const ScopedTransaction& trans );
    
    /// Default destructor
    virtual ~ScopedTransaction();

    /// Starts a new transaction. Returns the success of the operation
    bool start( bool readOnly=true );

    /// Commits the transaction.
    bool commit();

    /// Rolls back the transaction
    bool rollback();

    /// Checks if the transaction is active
    bool isActive( bool checkIfReadOnly=false ) const;

    private:
    Transaction& m_dbTransaction;
  };

}

#endif

