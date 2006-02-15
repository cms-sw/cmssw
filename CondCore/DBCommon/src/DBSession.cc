#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "FileCatalog/URIParser.h"
#include "FileCatalog/FCSystemTools.h"
#include "FileCatalog/IFileCatalog.h"
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/ITransaction.h"
#include "DataSvc/DataSvcFactory.h"
#include "DataSvc/IDataSvc.h"
#include "SealKernel/Exception.h"
cond::DBSession::DBSession( const std::string& con ):m_con(con),m_cat(new pool::IFileCatalog),m_svc( pool::DataSvcFactory::instance(m_cat)), m_catalogcon(""){}
cond::DBSession::DBSession( const std::string& con, 
			    const std::string& catalogcon )
  :m_con(con),m_cat(new pool::IFileCatalog),m_svc( pool::DataSvcFactory::instance(m_cat)), m_catalogcon(catalogcon){}
cond::DBSession::~DBSession(){
  delete m_cat;
  delete m_svc;
}
void cond::DBSession::setCatalog( const std::string& catalogCon ){
  m_catalogcon=catalogCon;
}
void cond::DBSession::connect( cond::ConnectMode mode ){
  pool::DatabaseConnectionPolicy policy;  
  switch(mode){
  case cond::ReadWriteCreate:
    policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
    policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::UPDATE);
    policy.setReadMode(pool::DatabaseConnectionPolicy::UPDATE);
  case cond::ReadWrite:
    policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::RAISE_ERROR);
    policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::UPDATE);
    policy.setReadMode(pool::DatabaseConnectionPolicy::UPDATE);
  case cond::ReadOnly:
    policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::RAISE_ERROR);
    policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::RAISE_ERROR);
    policy.setReadMode(pool::DatabaseConnectionPolicy::READ);
  default:
    throw cond::Exception(std::string("DBSession::connect unknown connect mode"));
  }
  pool::URIParser p;
  p.parse();
  m_cat->setWriteCatalog(p.contactstring());
  m_cat->connect();
  m_cat->start();
  m_svc->session().setDefaultConnectionPolicy(policy);
}
void cond::DBSession::disconnect(){
  m_svc->session().disconnectAll();
  m_cat->commit();
  m_cat->disconnect();
}
void cond::DBSession::startUpdate(){
  try{
    m_svc->session().transaction().start(pool::ITransaction::UPDATE);
  }catch(const seal::Exception& er){
    throw cond::Exception(std::string("DBSession::startUpdate caught seal::Exception ")+er.what());
  }catch(...){
    throw cond::Exception(std::string("DBSession::startUpdate caught unknown exception in "));
  }
}
void cond::DBSession::startReadOnly(){
  try{
    m_svc->session().transaction().start(pool::ITransaction::READ);
  }catch(const seal::Exception& er){
    throw cond::Exception(std::string("DBSession::startReadOnly caught seal::Exception ")+er.what());
  }catch(...){
    throw cond::Exception(std::string("DBSession::startReadOnly caught unknown exception in "));
  }
}
void cond::DBSession::rollback(){
  m_svc->session().transaction().rollback();
}
void cond::DBSession::commit(){
  try{
    m_svc->session().transaction().commit();
  }catch( const seal::Exception& er){
    throw cond::Exception( std::string("DBWriter::commit caught seal::Exception ")+ er.what() );
  }catch( ... ){
    throw cond::Exception( std::string("DBWriter::commit caught unknown exception ") );
  }
}
const std::string cond::DBSession::connectionString() const{
  return m_con;
}
pool::IDataSvc& cond::DBSession::DataSvc() const{
  return *m_svc;
}
pool::IFileCatalog& cond::DBSession::Catalog() const{
  return *m_cat;
}
