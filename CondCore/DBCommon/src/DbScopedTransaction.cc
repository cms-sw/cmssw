// local includes
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"

cond::DbScopedTransaction::DbScopedTransaction( cond::DbSession& session ):
  m_session(session),m_locallyActive(false){
}

cond::DbScopedTransaction::~DbScopedTransaction(){
  if(m_locallyActive) {
    m_session.transaction().rollback();
  }
}

int cond::DbScopedTransaction::start(bool readOnly){
  if(m_locallyActive) return -1;
  int ret = m_session.transaction().start( readOnly );
  m_locallyActive = true;
  return ret;
}

int cond::DbScopedTransaction::commit(){
  if(!m_locallyActive) return -1;
  int ret = m_session.transaction().commit();
  m_locallyActive = false;
  return ret;
}

bool cond::DbScopedTransaction::rollback(){
  if(!m_locallyActive) return false;
  bool ret = m_session.transaction().rollback();
  m_locallyActive = false;
  return ret;
}

bool cond::DbScopedTransaction::isLocallyActive() const {
  return m_locallyActive;
}

int cond::DbScopedTransaction::isActive() const {
  return m_session.transaction().isActive();
}

bool cond::DbScopedTransaction::isReadOnly() const 
{
  return m_session.transaction().isReadOnly();
}

