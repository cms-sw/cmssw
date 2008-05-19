#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "PoolConnectionProxy.h"
//pool includes
#include "DataSvc/IDataSvc.h"
#include "PersistencySvc/ITransaction.h"
//#include <iostream>
cond::PoolTransaction::PoolTransaction(cond::PoolConnectionProxy* parentConnection):m_parentConnection(parentConnection),m_datasvc(0), m_count(0){
  this->attach(m_parentConnection);
}

cond::PoolTransaction::~PoolTransaction(){}

void 
cond::PoolTransaction::start(bool readOnly){
  m_count++;
  if (1==m_count) {
    this->NotifyStartOfTransaction();
    if(!m_datasvc) throw cond::Exception("PoolTransaction::start: database not connected");
    if(!readOnly){
      m_datasvc->transaction().start( pool::ITransaction::UPDATE );
    }else{
      m_datasvc->transaction().start( pool::ITransaction::READ );
    }
    return;
  }
  if (!readOnly && isReadOnly()) upgrade();

}

void 
cond::PoolTransaction::commit(){
  if (0==m_count) return;
  m_count--;
  if (0==m_count) forceCommit();
}

void 
cond::PoolTransaction::upgrade() {
  forceCommit();
  m_datasvc->transaction().start( pool::ITransaction::UPDATE );
}

void  
cond::PoolTransaction::forceCommit() {
  if(!m_datasvc) throw cond::Exception("PoolTransaction::commit: database not connected");
  if(!m_datasvc->transaction().commit()){
    m_datasvc->transaction().rollback();
    throw cond::TransactionException("cond::PoolTransaction::commit","An Error ocurred, transaction rolled back");
  }
  this->NotifyEndOfTransaction();
}


void 
cond::PoolTransaction::rollback(){
  if(!m_datasvc) throw cond::Exception("PoolTransaction::rollback: database not connected");
   m_datasvc->transaction().rollback();
   this->NotifyEndOfTransaction();
}
bool 
cond::PoolTransaction::isReadOnly()const{
  if(m_datasvc->transaction().type()==pool::ITransaction::READ) return true;
  return false;
}
cond::IConnectionProxy& 
cond::PoolTransaction::parentConnection(){
  return *m_parentConnection;
}
void
cond::PoolTransaction::resetPoolDataSvc(pool::IDataSvc* datasvc) const{
  m_datasvc=datasvc;
}
pool::IDataSvc& 
cond::PoolTransaction::poolDataSvc(){
  return *m_datasvc;
}
