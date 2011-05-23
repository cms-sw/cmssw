// local includes
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/DbOpenTransaction.h"

cond::DbOpenTransaction::DbOpenTransaction( cond::DbTransaction& transaction ):
  m_transaction(transaction),m_status(false){
}

#include <iostream>
cond::DbOpenTransaction::~DbOpenTransaction(){
  if(!m_status && m_transaction.isActive() ) {
    m_transaction.rollback();
  }
}

void cond::DbOpenTransaction::ok()
{
  m_status = true;
}

