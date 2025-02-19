#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/ORA/interface/Transaction.h"

ora::ScopedTransaction::ScopedTransaction( Transaction& dbTransaction ):
  m_dbTransaction( dbTransaction ){
}

ora::ScopedTransaction::ScopedTransaction( const ScopedTransaction& trans ):
  m_dbTransaction( trans.m_dbTransaction ){
}

ora::ScopedTransaction::~ScopedTransaction() {
  if( m_dbTransaction.isActive() ) {
    rollback();
  }
}

bool ora::ScopedTransaction::start( bool readOnly ){
  return m_dbTransaction.start( readOnly );
}

bool ora::ScopedTransaction::commit(){
  return m_dbTransaction.commit( );
}

bool ora::ScopedTransaction::rollback(){
  return m_dbTransaction.rollback( );
}

bool ora::ScopedTransaction::isActive( bool checkIfReadOnly ) const {
  return m_dbTransaction.isActive( checkIfReadOnly );
}

