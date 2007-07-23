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
      bool isReadOnly,
      unsigned int connectionTimeOut):
  m_connectionSvcHandle(connectionServiceHandle),
  m_con(con),
  m_isReadOnly(isReadOnly),
  m_coralHandle(0),
  m_transactionCounter(0),
  m_connectionTimeOut(connectionTimeOut),
  m_transaction(new cond::CoralTransaction(this)){
}
cond::CoralConnectionProxy::~CoralConnectionProxy(){
}
cond::ITransaction&  
cond::CoralConnectionProxy::transaction(){
  return *m_transaction;
}
bool 
cond::CoralConnectionProxy::isReadOnly() const {
  return m_isReadOnly;
}
unsigned int
cond::CoralConnectionProxy::connectionTimeOut() const{
  return m_connectionTimeOut;
}
std::string 
cond::CoralConnectionProxy::connectStr() const{
  return m_con;
}
void 
cond::CoralConnectionProxy::connect(){
  m_coralHandle = m_connectionSvcHandle->connect(m_con, ( m_isReadOnly ) ? coral::ReadOnly : coral::Update );
  if(m_connectionTimeOut!=0){
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
    this->connect();
    static_cast<const cond::CoralTransaction*>(transactionSubject)->resetCoralHandle(m_coralHandle);
  }
  ++m_transactionCounter;
}
void 
cond::CoralConnectionProxy::reactOnEndOfTransaction( const cond::ITransaction* transactionSubject ){
  unsigned int connectedTime=(unsigned int)m_timer.elapsed();
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
