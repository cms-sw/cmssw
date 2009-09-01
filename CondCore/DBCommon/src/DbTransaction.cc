// local includes
#include "CondCore/DBCommon/interface/DbTransaction.h"
// coral & pool includes
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/ITransaction.h"

cond::DbTransaction::DbTransaction(coral::ISessionProxy& coralSession, pool::ISession& poolSession):
  m_coralSession(coralSession),m_poolSession(poolSession),m_readOnly(true),m_clients(0){
}

cond::DbTransaction::~DbTransaction(){
}

int cond::DbTransaction::start(bool readOnly){
  if(!m_clients){
    m_coralSession.transaction().start( readOnly );
    pool::ITransaction::Type transType = pool::ITransaction::READ;
    if(! readOnly ) transType = pool::ITransaction::UPDATE;
    m_poolSession.transaction().start( transType );
    m_readOnly = readOnly;
  } else {
    if(readOnly != m_readOnly ) return -1;
  }
  ++m_clients;
  return m_clients;
}

int cond::DbTransaction::commit(){
  if(!m_clients) return -1;
  else{
    --m_clients;
    if(m_clients == 0){
      // order matters!
      m_poolSession.transaction().commit();
      m_coralSession.transaction().commit();
    }
  }
  return m_clients;
}

bool cond::DbTransaction::forceCommit(){
  bool doCommit = false;
  if(m_clients){
    m_poolSession.transaction().commit();
    m_coralSession.transaction().commit();
    doCommit = true;
  }
  m_clients = 0;
  return doCommit;
}

bool cond::DbTransaction::rollback(){
  bool doRollBack = false;
  if(m_clients){
    m_poolSession.transaction().rollback();
    m_coralSession.transaction().rollback();
    doRollBack = true;
  }
  m_clients = 0;
  return doRollBack;
}

int cond::DbTransaction::isActive() const {
  if(!m_coralSession.transaction().isActive()) return 0;
  return m_clients;
}

bool cond::DbTransaction::isReadOnly() const 
{
  return m_coralSession.transaction().isReadOnly();
}

