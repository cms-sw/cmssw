#ifndef INCLUDE_ORA_TRANSACTION_H
#define INCLUDE_ORA_TRANSACTION_H

namespace ora {

  class DatabaseSession;
  
  /** @class Transaction Transaction.h 
   *
   */

  class Transaction {

    public:
    explicit Transaction( DatabaseSession& session );
    
    /// Default destructor
    virtual ~Transaction();

    /// Starts a new transaction. Returns the success of the operation
    bool start( bool readOnly=true );

    /// Commits the transaction.
    bool commit();

    /// Rolls back the transaction
    bool rollback();

    /// Checks if the transaction is active
    bool isActive( bool checkIfReadOnly=false ) const;
    private:
    DatabaseSession& m_session;
    bool m_localActive;
  };

}

#endif

