#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/Exception.h"
//#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "FileCatalog/IFileCatalog.h"
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/IDatabase.h"
#include "PersistencySvc/ITransaction.h"
#include "PersistencySvc/ITokenIterator.h"
#include "PersistencySvc/IContainer.h"
#include "POOLCore/Token.h"
#include "StorageSvc/DbType.h"
#include "DataSvc/DataSvcFactory.h"
#include "DataSvc/IDataSvc.h"
#include "DataSvc/RefBase.h"
#include "DataSvc/AnyPtr.h"
//#include "POOLCore/Exception.h"

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
  m_cat->start();
}
void cond::PoolStorageManager::disconnect(){
  m_svc->session().disconnectAll();
  m_cat->commit();
  m_cat->disconnect();
}
void cond::PoolStorageManager::startTransaction(bool isReadOnly){
  //m_cat->start();
  if(!isReadOnly){
    m_svc->transaction().start( pool::ITransaction::UPDATE );
  }else{
    m_svc->transaction().start( pool::ITransaction::READ );
  }
}
void cond::PoolStorageManager::commit(){
  m_svc->transaction().commit();
  //m_cat->commit();
}
void cond::PoolStorageManager::rollback(){
  m_svc->transaction().rollback();
  //m_cat->rollback();
}
std::string
cond::PoolStorageManager::copyObjectTo( cond::PoolStorageManager& destDB,
					const std::string& className,
					const std::string& objectToken ){
  const ROOT::Reflex::Type myclassType=ROOT::Reflex::Type::ByName(className);
  pool::RefBase myobj(m_svc,objectToken,myclassType.TypeInfo() );
  const pool::AnyPtr myPtr=myobj.object().get();
  std::string mycontainer=myobj.token()->contID();
  pool::Placement destPlace;
  destPlace.setDatabase(destDB.connectionString(), 
			pool::DatabaseSpecification::PFN );
  destPlace.setContainerName(mycontainer);
  destPlace.setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
  pool::RefBase mycopy(&(destDB.DataSvc()),myPtr,myclassType.TypeInfo());
  mycopy.markWrite(destPlace);
  //return token string of the copy
  return mycopy.toString();
}
void 
cond::PoolStorageManager::copyContainerTo( cond::PoolStorageManager& destDB,
					   const std::string& className,
					   const std::string& containerName ){
  const ROOT::Reflex::Type myclassType=ROOT::Reflex::Type::ByName(className);
  pool::Placement destPlace;
  destPlace.setDatabase(destDB.connectionString(), 
			pool::DatabaseSpecification::PFN );
  destPlace.setContainerName(containerName);
  destPlace.setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
  pool::ITokenIterator* tokenIt=m_svc->session().databaseHandle(m_con,pool::DatabaseSpecification::PFN)->containerHandle( containerName )->tokens("");
  pool::Token* myToken=0;
  while( (myToken=tokenIt->next())!=0 ){
    pool::RefBase myobj(m_svc,*myToken,myclassType.TypeInfo() );
    const pool::AnyPtr myPtr=myobj.object().get();
    pool::RefBase mycopy(&(destDB.DataSvc()),myPtr,myclassType.TypeInfo());
    mycopy.markWrite(destPlace);
    myToken->release();
  }
  delete tokenIt;//?????what about others?
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
