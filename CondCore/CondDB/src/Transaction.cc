#include "CondCore/CondDB/interface/Transaction.h"

namespace cond {

  namespace db  {

    TransactionScope::TransactionScope( Transaction& transaction ):
      m_transaction(transaction),m_status(false){
    }

    TransactionScope::~TransactionScope(){
      if(!m_status && m_transaction.isActive() ) {
	m_transaction.rollback();
      }
    }
    
    void TransactionScope::close()
    {
      m_status = true;
    }

  }

}
