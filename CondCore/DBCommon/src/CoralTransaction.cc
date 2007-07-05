//local includes
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CoralConnectionProxy.h"
//coral includes
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"

cond::CoralTransaction::CoralTransaction(cond::CoralConnectionProxy* parentConnection):m_parentConnection(parentConnection){
  
}
cond::CoralTransaction::~CoralTransaction(){}
void
cond::CoralTransaction::resetCoralHandle(coral::ISessionProxy* coralHandle) const{
  m_coralHandle=coralHandle;
}
void 
cond::CoralTransaction::start(bool isReadOnly){
  if(!m_coralHandle) throw cond::Exception("CoralTransaction::start database not connected");
  m_coralHandle->transaction().start(isReadOnly);
  this->NotifyStartOfTransaction();
  m_isReadOnly=isReadOnly;
}
void 
cond::CoralTransaction::commit(){
  if(!m_coralHandle) throw cond::Exception("CoralTransaction::commit database not connected");
  m_coralHandle->transaction().commit();
  this->NotifyEndOfTransaction();
}
void 
cond::CoralTransaction::rollback(){
  if(!m_coralHandle) throw cond::Exception("CoralTransaction::rollback database not connected");
  m_coralHandle->transaction().rollback();
  this->NotifyEndOfTransaction();
}
bool 
cond::CoralTransaction::isReadOnly()const{
  return m_isReadOnly;
}
cond::IConnectionProxy& 
cond::CoralTransaction::parentConnection(){
  return *m_parentConnection;
}
