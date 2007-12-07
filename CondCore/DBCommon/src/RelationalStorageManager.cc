#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include <string>
#include "ServiceLoader.h"
//#include "SealKernel/ComponentLoader.h"
#include "SealKernel/Component.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "CoralBase/Exception.h"
#include <vector>
cond::RelationalStorageManager::RelationalStorageManager(const std::string& con):m_con(con),m_proxy(0),m_started(false),m_sessionHandle(new cond::DBSession(false)), m_sessionShared(false){
}
cond::RelationalStorageManager::RelationalStorageManager(const std::string& con, cond::DBSession* session ):m_con(con),m_proxy(0),m_started(false),m_sessionHandle(session),m_sessionShared(true){}
cond::RelationalStorageManager::~RelationalStorageManager(){
  if(!m_sessionShared) delete m_sessionHandle;
}
cond::DBSession& cond::RelationalStorageManager::session(){
  return *m_sessionHandle;
}
bool cond::RelationalStorageManager::isSessionShared() const{
  return m_sessionShared;
} 
coral::ISessionProxy* cond::RelationalStorageManager::connect(cond::ConnectMode mod){
  if(!m_started) init();
  if(mod==cond::ReadOnly){
    m_readOnlyMode=true;
  }
  if ( m_proxy ) {
    delete m_proxy;
    m_proxy= 0;
  }
  try{
    m_proxy = connectionService()->connect(m_con, ( mod == cond::ReadOnly ) ? coral::ReadOnly : coral::Update );
  }catch (const coral::Exception& e) {
    if(m_proxy) delete m_proxy;
    m_proxy = 0;
    throw cond::Exception( std::string("RelationalStorageManager::connect: couldn't connect to the database ")+e.what() );
  }
  return m_proxy;
}
void cond::RelationalStorageManager::disconnect(){
  try{
    coral::ITransaction& transaction = m_proxy->transaction();
    //if ( transaction.isActive() && ( ! m_proxy->isConnectionShared() ) ) {
    if ( transaction.isActive() ) {
      if ( transaction.isReadOnly() ) {
	//if ( m_proxy->isConnectionShared() ) {
	transaction.commit();
	//}
      }else {
	transaction.rollback();
      }
    }
  }catch(const coral::Exception&){
  }
  if ( m_proxy ) delete m_proxy;
  m_proxy = 0;
}
void cond::RelationalStorageManager::startTransaction(bool isReadOnly){
  coral::ITransaction& transaction = m_proxy->transaction();
  //bool sharedConnection = m_proxy->isConnectionShared();
  bool activeTransaction = transaction.isActive();
  try {
    if(!activeTransaction) transaction.start(isReadOnly);
  }catch(const coral::Exception& e){
    if(m_proxy) delete m_proxy;
    m_proxy = 0;
    throw cond::Exception( std::string("RelationalStorageManager::startTransaction:")+e.what() );
  }
}
void cond::RelationalStorageManager::commit(){
  coral::ITransaction& transaction = m_proxy->transaction();
  // bool sharedConnection = m_proxy->isConnectionShared();
  bool activeTransaction = transaction.isActive();
  try {
    //if(!activeTransaction && ! sharedConnection ) {
    if(!activeTransaction ) {
      if(m_proxy) delete m_proxy;
      m_proxy = 0;
      throw cond::Exception("RelationalStorageManager::commit cannot commit inactive transaction");
    }
    //if ( ! sharedConnection ) 
    transaction.commit();
  }catch(const coral::Exception& e){
    if(m_proxy) delete m_proxy;
    m_proxy = 0;
    throw cond::Exception( std::string("RelationalStorageManager::commit:")+e.what() );
  }
}
void cond::RelationalStorageManager::rollback(){
  coral::ITransaction& transaction = m_proxy->transaction();
  //bool sharedConnection = m_proxy->isConnectionShared();
  bool activeTransaction = transaction.isActive();
  if ( activeTransaction ) {
    //if ( ! sharedConnection ) 
    transaction.rollback();
  }
}
std::string cond::RelationalStorageManager::connectionString() const{
  return m_con;
}
coral::ISessionProxy& cond::RelationalStorageManager::sessionProxy(){
  return *m_proxy;
}
void cond::RelationalStorageManager::init(){
  if( !m_sessionHandle->isActive()){
    m_sessionHandle->open();
  }
  m_started=true;
}

seal::IHandle<coral::IConnectionService>
cond::RelationalStorageManager::connectionService()
{
  std::vector< seal::IHandle<coral::IConnectionService> > v_svc;
  m_sessionHandle->serviceLoader().context()->query( v_svc );
  if ( v_svc.empty() ) {
    throw cond::Exception( "Could not locate the connection service" );
  }
  return v_svc.front();
}
