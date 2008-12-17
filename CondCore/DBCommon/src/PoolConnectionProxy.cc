//local includes
#include "PoolConnectionProxy.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
//connection service includes
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "CoralKernel/Context.h"
#include "CoralKernel/IHandle.h"
//pool includes
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/IConfiguration.h"
#include "PersistencySvc/ITransaction.h"
#include "DataSvc/DataSvcFactory.h"
#include "DataSvc/IDataSvc.h"
#include "FileCatalog/IFileCatalog.h"
#include "POOLCore/IBlobStreamingService.h"
//#include <iostream>
cond::PoolConnectionProxy::PoolConnectionProxy(
	  coral::IConnectionService* connectionServiceHandle,
	  pool::IBlobStreamingService* blobStreamingServiceHandle,
	  const std::string& con,
	  int connectionTimeOut,
	  int idleConnectionCleanupPeriod):
  cond::IConnectionProxy(connectionServiceHandle,con,connectionTimeOut,idleConnectionCleanupPeriod),
  m_blobstreamingService(blobStreamingServiceHandle),
  m_datasvc(0),
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
  // disconnect(); (at the moment crashes)
  m_catalog->commit();
  m_catalog->disconnect();
  //m_datasvc->session().disconnectAll();
  if(m_transaction) delete m_transaction;
  //delete m_datasvc;
  delete m_catalog;
  //m_datasvc=0;
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
  if(!m_datasvc){
    m_datasvc=pool::DataSvcFactory::instance(m_catalog);
  }
  m_datasvc->configuration().setConnectionService(m_connectionSvcHandle,false);
  if(m_blobstreamingService!=0){
    m_datasvc->configuration().setBlobStreamer(m_blobstreamingService, false);
  }
  
  pool::DatabaseConnectionPolicy policy;
  policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
  policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::UPDATE);
  policy.setReadMode(pool::DatabaseConnectionPolicy::READ);
  m_datasvc->session().setDefaultConnectionPolicy(policy);
  
  /*m_datasvc=pool::DataSvcFactory::instance(m_catalog);
  pool::DatabaseConnectionPolicy policy;
  policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
  policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::UPDATE);
  policy.setReadMode(pool::DatabaseConnectionPolicy::READ);
  m_datasvc->session().setDefaultConnectionPolicy(policy);
  */
  //if(m_connectionTimeOut>0){
  m_timer.restart();
  //}
}
void
cond::PoolConnectionProxy::disconnect(){
  if (0==m_datasvc) return;
  m_datasvc->transaction().commit();
  m_datasvc->session().disconnectAll();
  delete m_datasvc;
  m_datasvc=0;
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
  if(!m_connectionSvcHandle) throw cond::Exception("PoolConnectionProxy::reactOnStartOfTransaction: cannot start transaction database is not connected.");
  int connectedTime=(int)m_timer.elapsed();
  //std::cout<<"pool connectedTime "<<connectedTime<<std::endl;
  //std::cout<<"isPoolAutimaticCleanUpEnabled() "<<m_connectionSvcHandle->configuration().isPoolAutomaticCleanUpEnabled()<<std::endl;
  if(!m_connectionSvcHandle->configuration().isPoolAutomaticCleanUpEnabled()){
    //std::cout<<"idlepoolcleanupPeriod "<<m_idleConnectionCleanupPeriod<<std::endl;
    //std::cout<<"pool connected time "<<connectedTime<<std::endl;
    if(connectedTime>=m_idleConnectionCleanupPeriod){
      m_connectionSvcHandle->purgeConnectionPool();
      //std::cout<<"idle connection pool purged"<<std::endl;
    }
  }
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
