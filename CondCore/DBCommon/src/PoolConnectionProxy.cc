//local includes
#include "PoolConnectionProxy.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
//connection service includes
#include "RelationalAccess/IConnectionService.h"
//pool includes
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/ITransaction.h"
#include "DataSvc/DataSvcFactory.h"
#include "DataSvc/IDataSvc.h"
#include "FileCatalog/IFileCatalog.h"
//#include <iostream>
cond::PoolConnectionProxy::PoolConnectionProxy(
	  coral::IConnectionService* connectionServiceHandle,
	  const std::string& con,
	  int connectionTimeOut):
  cond::IConnectionProxy(connectionServiceHandle,con,connectionTimeOut),
  m_transaction( 0 ),
  m_transactionCounter( 0 ),
  m_catalog( new pool::IFileCatalog ) 
{
  std::string catconnect("pfncatalog_memory://POOL_RDBMS?");
  catconnect.append(con);
  m_catalog->setWriteCatalog(catconnect);
  m_catalog->connect();
  m_catalog->start();
}
cond::PoolConnectionProxy::~PoolConnectionProxy(){
  //std::cout<<"PoolConnectionProxy::~PoolConnectionProxy"<<std::endl;
  m_catalog->commit();
  m_catalog->disconnect();
  //m_datasvc->session().disconnectAll();
  delete m_transaction;
  delete m_datasvc;
  delete m_catalog;
  m_datasvc=0;
}
cond::ITransaction&  
cond::PoolConnectionProxy::transaction(){
  if(!m_transaction){
    m_transaction=new cond::PoolTransaction(this);
  }
  return *m_transaction;
}
pool::IDataSvc* 
cond::PoolConnectionProxy::poolDataSvc(){
  return m_datasvc;
}
void 
cond::PoolConnectionProxy::connect(){
  m_datasvc=pool::DataSvcFactory::instance(m_catalog);
  pool::DatabaseConnectionPolicy policy;
  policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
  policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::UPDATE);
  policy.setReadMode(pool::DatabaseConnectionPolicy::READ);
  m_datasvc->session().setDefaultConnectionPolicy(policy);
  if(m_connectionTimeOut>0){
    m_timer.restart();
  }
}
void
cond::PoolConnectionProxy::disconnect(){
  m_datasvc->transaction().commit();
  m_datasvc->session().disconnectAll();
}
void 
cond::PoolConnectionProxy::reactOnStartOfTransaction( const ITransaction* transactionSubject ){
  if(m_transactionCounter==0){
    this->connect();
    static_cast<const cond::PoolTransaction*>(transactionSubject)->resetPoolDataSvc(m_datasvc);
  }
  ++m_transactionCounter;
}
void 
cond::PoolConnectionProxy::reactOnEndOfTransaction( const ITransaction* transactionSubject ){  
  if(m_connectionTimeOut==0){
    this->disconnect();
  }else{
    unsigned int connectedTime=(unsigned int)m_timer.elapsed();
    if(m_transactionCounter==1 && connectedTime>= m_connectionTimeOut){
      //if I'm the last open transaction and I'm beyond the connection timeout, close connection
      this->disconnect();
    }
  }
  --m_transactionCounter;
}
