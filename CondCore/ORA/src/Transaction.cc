#include "CondCore/ORA/interface/Transaction.h"
#include "DatabaseSession.h"

ora::Transaction::Transaction(DatabaseSession& session):
  m_session( session ),
  m_localActive( false ){
}

ora::Transaction::~Transaction() {
}

bool ora::Transaction::start( bool readOnly ){
  bool started = false;
  if(!m_localActive){
    m_session.startTransaction( readOnly );
    m_localActive = true;
    started = true;
  }
  return started;
}

bool ora::Transaction::commit(){
  bool committed = false;
  if(m_localActive){
    m_session.commitTransaction();
    m_localActive = false;
    committed = true;
  }
  return committed;
}

bool ora::Transaction::rollback(){
  bool rolled = false;
  if(m_localActive){
    m_session.rollbackTransaction();
    m_localActive = false;
    rolled = true;
  }
  return rolled;
}

bool ora::Transaction::isActive() const {
  return m_session.isTransactionActive();
}

