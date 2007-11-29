#include "CoralConnectionProxy.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
//coral includes
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/AccessMode.h"

//#include <iostream>
cond::CoralConnectionProxy::CoralConnectionProxy(
      coral::IConnectionService* connectionServiceHandle,
      const std::string& con,
      int connectionTimeOut):
  cond::IConnectionProxy(connectionServiceHandle,con,connectionTimeOut),
  m_transactionCounter(0),
  m_transaction(new cond::CoralTransaction(this)){
}
cond::CoralConnectionProxy::~CoralConnectionProxy(){
}
cond::ITransaction&  
cond::CoralConnectionProxy::transaction(){
  return *m_transaction;
}
void 
cond::CoralConnectionProxy::connect(bool isReadOnly){
  m_coralHandle = m_connectionSvcHandle->connect(m_con, ( isReadOnly ) ? coral::ReadOnly : coral::Update );
  if(m_connectionTimeOut>0){
    m_timer.restart();
  }
}
void
cond::CoralConnectionProxy::disconnect(){
  if(m_coralHandle){
    delete m_coralHandle;
    m_coralHandle=0;
  }
}
coral::ISessionProxy&
cond::CoralConnectionProxy::coralProxy(){
  return *m_coralHandle;
}
void 
cond::CoralConnectionProxy::reactOnStartOfTransaction( const cond::ITransaction* transactionSubject ){
  if(m_transactionCounter==0){    
    this->connect(transactionSubject->isReadOnly());
    static_cast<const cond::CoralTransaction*>(transactionSubject)->resetCoralHandle(m_coralHandle);
  }
  ++m_transactionCounter;
}
void 
cond::CoralConnectionProxy::reactOnEndOfTransaction( const cond::ITransaction* transactionSubject ){
  int connectedTime=(int)m_timer.elapsed();
  if(m_connectionTimeOut==0){
    this->disconnect();
  }else{
    if(m_transactionCounter==1 && connectedTime>= m_connectionTimeOut){
      //if I'm the last open transaction and I'm beyond the connection timeout, close connection
      this->disconnect();
    }
  }
  --m_transactionCounter;
}
