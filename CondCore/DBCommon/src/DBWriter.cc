#include "CondCore/DBCommon/interface/DBWriter.h"
//#include "PluginManager/PluginManager.h"
#include "FileCatalog/URIParser.h"
#include "FileCatalog/FCSystemTools.h"
#include "FileCatalog/IFileCatalog.h"
#include "StorageSvc/DbType.h"
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/ITransaction.h"
#include "PersistencySvc/IDatabase.h"
#include "PersistencySvc/Placement.h"
#include "DataSvc/DataSvcFactory.h"
#include "DataSvc/IDataSvc.h"
#include "DataSvc/Ref.h"
#include "POOLCore/POOLContext.h"
#include "SealKernel/Exception.h"
#include <algorithm>

cond::DBWriter::DBWriter( const std::string& con ):m_con(con),m_cat(new pool::IFileCatalog),m_svc( pool::DataSvcFactory::instance(m_cat) ), m_placement(new pool::Placement){
  //seal::PluginManager::get()->initialise();//should not be called by me!!
  //pool::POOLContext::loadComponent( "POOL/Services/EnvironmentAuthenticationService" );
  pool::URIParser p;
  p.parse();
  m_cat->setWriteCatalog(p.contactstring());
  m_cat->connect();
  m_cat->start();
  m_placement->setTechnology(pool::POOL_RDBMS_StorageType.type());
}

cond::DBWriter::~DBWriter(){
  m_cat->commit();
  m_cat->disconnect();
  delete m_cat;
  delete m_svc;
  delete m_placement;
}

void cond::DBWriter::startTransaction(){
  /* should add policy parameters*/
  m_svc->transaction().start(pool::ITransaction::UPDATE);
}

void cond::DBWriter::commitTransaction(){
  try{
    m_svc->transaction().commit();
    m_svc->session().disconnectAll();
  }catch( const seal::Exception& er){
    std::cout << er.what() << std::endl;    
  }catch ( const std::exception& er ) {
    std::cout << er.what() << std::endl;
  }catch ( ... ) {
    std::cout << "Funny error" << std::endl;
  }
}

bool cond::DBWriter::containerExists(const std::string& containerName){
    pool::ITransaction& transaction = m_svc->session().transaction();
    transaction.start( pool::ITransaction::READ );
    std::auto_ptr< pool::IDatabase > db( m_svc->session().databaseHandle( m_con,pool::DatabaseSpecification::PFN ) );
    try{
      db->connectForRead();
    }catch( const seal::Exception& er){
      std::cout << er.what() << std::endl;    
      transaction.commit();
      return false;
    }catch ( const std::exception& er ) {
      std::cout << er.what() << std::endl;
      transaction.commit();
      return false;
    }catch ( ... ) {
      std::cout << "Funny error" << std::endl;
      transaction.commit();
      return false;
    }
    std::vector< std::string > containers = db->containers();
    transaction.commit();
    std::vector<std::string>::const_iterator i=std::find(containers.begin(), containers.end(), containerName);
    return (i!=containers.end());   
}

void cond::DBWriter::openContainer( const std::string& containerName ){
  m_placement->setContainerName(containerName);
  pool::DatabaseConnectionPolicy policy;
  policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::UPDATE); 
  m_svc->session().setDefaultConnectionPolicy(policy);
  m_placement->setDatabase(m_con, pool::DatabaseSpecification::PFN); 
}

void cond::DBWriter::createContainer( const std::string& containerName ){
  m_placement->setContainerName(containerName);
  pool::DatabaseConnectionPolicy policy;  
  policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
  policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::UPDATE); 
  m_svc->session().setDefaultConnectionPolicy(policy);
  m_placement->setDatabase(m_con, pool::DatabaseSpecification::PFN); 
}
