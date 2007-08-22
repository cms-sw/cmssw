//local includes
#include "PoolConnectionProxy.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
//pool includes
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/ITransaction.h"
#include "DataSvc/DataSvcFactory.h"
#include "DataSvc/IDataSvc.h"
#include "FileCatalog/IFileCatalog.h"
cond::PoolConnectionProxy::PoolConnectionProxy(const std::string& con,
					       const std::string& catalog,
					       bool isReadOnly,
					       unsigned int connectionTimeOut):
  m_datasvc( 0 ),
  m_transaction( 0 ),
  m_con( con ),
  m_catalog( new pool::IFileCatalog ),
  m_isReadOnly( isReadOnly ),
  m_transactionCounter( 0 ),
  m_connectionTimeOut( connectionTimeOut )
{
  m_catalog->setWriteCatalog(catalog);
  //if(!m_isReadOnly){
  //}else{
  //  m_catalog->addReadCatalog(catalog);
  //}
}
cond::PoolConnectionProxy::~PoolConnectionProxy(){
  delete m_transaction;
  delete m_datasvc;
  m_datasvc=0;
}
cond::ITransaction&  
cond::PoolConnectionProxy::transaction(){
  if(!m_transaction){
    m_transaction=new cond::PoolTransaction(this);
  }
  return *m_transaction;
}
bool 
cond::PoolConnectionProxy::isReadOnly() const {
  return m_isReadOnly;
}
unsigned int
cond::PoolConnectionProxy::connectionTimeOut() const{
  return m_connectionTimeOut;
}
std::string 
cond::PoolConnectionProxy::connectStr() const{
  return m_con;
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
  m_catalog->connect();
  m_catalog->start();
  if(m_connectionTimeOut!=0){
    m_timer.restart();
  }
  
}
void
cond::PoolConnectionProxy::disconnect(){
  m_datasvc->transaction().commit();
  m_datasvc->session().disconnectAll();
  m_catalog->commit();
  m_catalog->disconnect();
  //delete m_datasvc;
  //m_datasvc=0;
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
