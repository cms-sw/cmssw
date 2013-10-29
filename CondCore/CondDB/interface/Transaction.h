#ifndef CondCore_CondDB_Transaction_h
#define CondCore_CondDB_Transaction_h

// Package:     CondDB
// Class  :     Session
// 
/**CondCore/CondDB/interface/Transaction.h
   Description: DB Transaction related utilities 
*/
//
// Author:      Giacomo Govi
// Created:     Oct 2013

#include "CondCore/CondDB/interface/CondDB.h"

namespace cond {
  namespace db {

    class TransactionScope {
    public:
      explicit TransactionScope( Transaction& transaction );   
    
      ~TransactionScope();
      
      void close();
    private:
      Transaction& m_transaction;
      bool m_status;
      
    };
    
  }
}

#endif // CondCore_CondDB_Transaction_h
