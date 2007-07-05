#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "PoolConnectionProxy.h"
//pool includes
#include "DataSvc/IDataSvc.h"
#include "PersistencySvc/ITransaction.h"
cond::PoolTransaction::PoolTransaction(cond::PoolConnectionProxy* parentConnection):m_parentConnection(parentConnection){
  
}
cond::PoolTransaction::~PoolTransaction(){}
void 
cond::PoolTransaction::start(bool isReadOnly){
  if(!m_datasvc) throw cond::Exception("PoolTransaction::start: database not connected");
  if(!isReadOnly){
    m_datasvc->transaction().start( pool::ITransaction::UPDATE );
  }else{
    m_datasvc->transaction().start( pool::ITransaction::READ );
  }
  this->NotifyStartOfTransaction();
  m_isReadOnly=isReadOnly;
}
void 
cond::PoolTransaction::commit(){
  if(!m_datasvc) throw cond::Exception("PoolTransaction::commit: database not connected");
  m_datasvc->transaction().commit();
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
  return m_isReadOnly;
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
