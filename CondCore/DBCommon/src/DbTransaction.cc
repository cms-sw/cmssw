// local includes
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/ORA/interface/Transaction.h"
// coral & pool includes
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"

cond::DbTransaction::DbTransaction( ora::Transaction& dbTrans ):
  m_dbTrans( dbTrans ),m_readOnly(true),m_clients(0){
}

cond::DbTransaction::~DbTransaction(){
  rollback();
}

int cond::DbTransaction::start(bool readOnly){
  if(!m_clients){
    m_dbTrans.start( readOnly );
    m_readOnly = readOnly;
  } else {
    if(readOnly != m_readOnly)
      return -1;
  }
  ++m_clients;
  return m_clients;
}

int cond::DbTransaction::commit(){
  if(!m_clients) return -1;
  else{
    --m_clients;
    if(m_clients == 0){
      m_dbTrans.commit();      
    }
  }
  return m_clients;
}

bool cond::DbTransaction::forceCommit(){
  bool doCommit = false;
  if(m_clients){
    m_dbTrans.commit();
    doCommit = true;
  }
  m_clients = 0;
  return doCommit;
}

bool cond::DbTransaction::rollback(){
  bool doRollBack = false;
  if(m_clients){
    m_dbTrans.rollback();
    doRollBack = true;
  }
  m_clients = 0;
  return doRollBack;
}

int cond::DbTransaction::isActive() const {
  if(!m_dbTrans.isActive()) return 0;
  return m_clients;
}

bool cond::DbTransaction::isReadOnly() const 
{
  return m_dbTrans.isActive( true );
}

