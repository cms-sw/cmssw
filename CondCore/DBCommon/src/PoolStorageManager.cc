#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
//#include "FileCatalog/URIParser.h"
//#include "FileCatalog/FCSystemTools.h"
#include "FileCatalog/IFileCatalog.h"
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/IDatabase.h"
#include "PersistencySvc/ITransaction.h"
#include "DataSvc/DataSvcFactory.h"
#include "DataSvc/IDataSvc.h"
#include "POOLCore/Exception.h"
cond::PoolStorageManager::PoolStorageManager(const std::string& con,
					     const std::string& catalog): m_catalogstr(catalog),m_con(con),m_cat(new pool::IFileCatalog),m_svc( pool::DataSvcFactory::instance(m_cat)),m_db(0),m_started(false),m_sessionHandle(new cond::DBSession(true)),m_sessionShared(false){  
}
cond::PoolStorageManager::PoolStorageManager(const std::string& con,
					     const std::string& catalog,
					     cond::DBSession* session ): m_catalogstr(catalog),m_con(con),m_cat(new pool::IFileCatalog),m_svc( pool::DataSvcFactory::instance(m_cat)),m_db(0),m_started(false),m_sessionHandle(session){
  if( !m_sessionHandle ){
    m_sessionHandle=new cond::DBSession(true);
    m_sessionShared=false;
  }else{
    m_sessionShared=true;
  }
}
cond::PoolStorageManager::~PoolStorageManager(){
  delete m_cat;
  delete m_svc;
  if(m_db) delete m_db;
  if(!m_sessionShared) delete m_sessionHandle;
}
cond::DBSession& cond::PoolStorageManager::session(){
  return *m_sessionHandle;
}
bool cond::PoolStorageManager::isSessionShared() const{
  return m_sessionShared;
} 
void cond::PoolStorageManager::connect(){
  if(!m_started) init();
  m_cat->connect();
}
void cond::PoolStorageManager::disconnect(){
  m_svc->session().disconnectAll();
  m_cat->disconnect();
}
void cond::PoolStorageManager::startTransaction(bool isReadOnly){
  m_cat->start();
  if(!isReadOnly){
    m_svc->transaction().start( pool::ITransaction::UPDATE );
  }else{
    m_svc->transaction().start( pool::ITransaction::READ );
  }
}
void cond::PoolStorageManager::commit(){
  m_svc->transaction().commit();
  m_cat->commit();
}
void cond::PoolStorageManager::rollback(){
  m_svc->transaction().rollback();
  m_cat->rollback();
}
std::string cond::PoolStorageManager::catalogString() const{
  return m_catalogstr;
}
std::string cond::PoolStorageManager::connectionString() const{
  return m_con;
}
std::vector<std::string> cond::PoolStorageManager::containers(){
  if(!m_db){
    m_db=m_svc->session().databaseHandle(m_con, pool::DatabaseSpecification::PFN);
    if(!m_db){ throw cond::Exception( "PoolStorageManager::containers could not obtain database handle" ); }
  }
  return m_db->containers();
}
pool::IDataSvc& cond::PoolStorageManager::DataSvc(){
  return *m_svc;
}
void cond::PoolStorageManager::init(){
  if( !m_sessionHandle->isActive() ){
    m_sessionHandle->open();
  }
  pool::DatabaseConnectionPolicy policy;
  policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
  policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::UPDATE);
  policy.setReadMode(pool::DatabaseConnectionPolicy::READ);
  m_cat->setWriteCatalog(m_catalogstr);
  m_svc->session().setDefaultConnectionPolicy(policy);
  m_started=true;
}
