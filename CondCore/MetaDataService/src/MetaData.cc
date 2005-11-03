#include "CondCore/MetaDataService/interface/MetaData.h"
#include "RelationalAccess/RelationalException.h"
#include "RelationalAccess/IRelationalService.h"
#include "RelationalAccess/IRelationalDomain.h"
#include "RelationalAccess/IRelationalSession.h"
#include "RelationalAccess/IRelationalTransaction.h"
#include "RelationalAccess/IRelationalSchema.h"
#include "RelationalAccess/IRelationalTable.h"
#include "RelationalAccess/RelationalEditableTableDescription.h"
#include "RelationalAccess/IRelationalTablePrivilegeManager.h"
#include "RelationalAccess/IRelationalPrimaryKey.h"
#include "RelationalAccess/IRelationalCursor.h"
#include "RelationalAccess/IRelationalQuery.h"
#include "RelationalAccess/IRelationalTableDataEditor.h"
#include "AttributeList/AttributeList.h"
#include "POOLCore/POOLContext.h"
#include "SealKernel/Context.h"
#include "SealKernel/Service.h"
#include <stdexcept>

cond::MetaData::MetaData(const std::string& connectionString):m_con(connectionString),m_table(0){
  pool::POOLContext::loadComponent( "SEAL/Services/MessageService" );
  pool::POOLContext::loadComponent( "POOL/Services/RelationalService" );
  pool::POOLContext::loadComponent( "POOL/Services/EnvironmentAuthenticationService" );
  pool::POOLContext::setMessageVerbosityLevel( seal::Msg::Debug );
  m_log.reset(new seal::MessageStream(pool::POOLContext::context(), "MetaDataService" ));  
  seal::IHandle<pool::IRelationalService> serviceHandle=pool::POOLContext::context()->query<pool::IRelationalService>( "POOL/Services/RelationalService" );
  if ( ! serviceHandle ) {
    //throw std::runtime_error( "Could not retrieve the relational service" );
    (*m_log)<<seal::Msg::Error<<"cond::MetaData::MetaData: Could not retrieve the relational service"<<seal::flush;
  }
  pool::IRelationalDomain& domain = serviceHandle->domainForConnection(m_con);
  m_session.reset( domain.newSession( m_con ) );
  if ( ! m_session->connect() ) {
    //throw std::runtime_error( "Could not connect to the database server." );
    (*m_log)<<seal::Msg::Error<<"cond::MetaData::MetaData: Could not connect to the database"<<seal::flush;
  }
  if ( ! m_session->transaction().start() ) {
    (*m_log)<<seal::Msg::Error<<"cond::MetaData::MetaData: Could not start a new transaction"<<seal::flush;
  }
}

cond::MetaData::~MetaData(){
  (*m_log)<<seal::Msg::Debug<< "Disconnecting..." << seal::flush;
  if ( ! m_session->transaction().commit() ) {
    (*m_log)<<seal::Msg::Error<<"cond::MetaData::MetaData: Could not commit the transaction"<<seal::flush;
    //      throw std::runtime_error( "Could not commit the transaction." );
  }
  m_session->disconnect();
}

bool cond::MetaData::addMapping(const std::string& name, const std::string& iovtoken){
  (*m_log)<<seal::Msg::Debug<<"cond::MetaData::addMapping"<<seal::flush;
  std::string tablename("METADATA");
  if(!m_session->userSchema().existsTable(tablename)){
    this->createTable(tablename);
  }else{
    m_table=&(m_session->userSchema().tableHandle("METADATA"));
  }
  pool::AttributeList data( m_table->description().columnNamesAndTypes() );
  pool::IRelationalTableDataEditor& dataEditor = m_table->dataEditor();
  data["name"].setValue<std::string>(name);
  data["token"].setValue<std::string>(iovtoken);
  return( dataEditor.insertNewRow( data ) );
}

const std::string cond::MetaData::getToken( const std::string& name ){
  (*m_log)<<seal::Msg::Debug<<"cond::MetaData::getToken "<<name<<seal::flush;
  if(!m_table){
    m_table=&(m_session->userSchema().tableHandle("METADATA"));
  }
  std::string iovtoken;
  std::auto_ptr< pool::IRelationalQuery > query( m_table->createQuery() );
  query->setRowCacheSize( 10 );
  pool::AttributeList emptyBindVariableList;
  std::string condition=std::string("name='")+name+"'";
  query->setCondition( condition, emptyBindVariableList );
  query->addToOutputList( "token" );
  pool::IRelationalCursor& cursor = query->process();
  if ( cursor.start() ) {
    while( cursor.next() ) {
      const pool::AttributeList& row = cursor.currentRow();
      for ( pool::AttributeList::const_iterator iColumn = row.begin();
	    iColumn != row.end(); ++iColumn ) {
	std::cout << iColumn->spec().name() << " : " << iColumn->getValueAsString() << "\t";
	iovtoken=iColumn->getValueAsString();
      }
      std::cout << std::endl;
    }
  }
  return iovtoken;
}
void cond::MetaData::createTable(const std::string& tabname){
  pool::IRelationalSchema& schema=m_session->userSchema();
  seal::IHandle<pool::IRelationalService> serviceHandle=pool::POOLContext::context()->query<pool::IRelationalService>( "POOL/Services/RelationalService" );
  pool::IRelationalDomain& domain = serviceHandle->domainForConnection(m_con);
  std::auto_ptr< pool::IRelationalEditableTableDescription > desc( new pool::RelationalAccess::RelationalEditableTableDescription( *m_log, domain.flavorName() ) );
  desc->insertColumn( "name", pool::AttributeStaticTypeInfo<std::string>::type_name() );
  desc->insertColumn( "token", pool::AttributeStaticTypeInfo<std::string>::type_name() );
  std::vector<std::string> cols;
  cols.push_back("token");
  desc->setPrimaryKey(cols);
  desc->setNotNullConstraint( "name" );
  m_table=&(schema.createTable(tabname,*desc));  
  m_table->privilegeManager().grantToPublic( pool::IRelationalTablePrivilegeManager::SELECT );
}
