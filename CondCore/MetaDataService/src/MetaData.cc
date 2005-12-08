#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/MetaDataService/interface/MetaDataNames.h"
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
#include "FWCore/Utilities/interface/Exception.h"
cond::MetaData::MetaData(const std::string& connectionString):m_con(connectionString),m_table(0){
  pool::POOLContext::loadComponent( "POOL/Services/RelationalService" );
  m_log.reset(new seal::MessageStream(pool::POOLContext::context(), "MetaDataService" ));  
  seal::IHandle<pool::IRelationalService> serviceHandle=pool::POOLContext::context()->query<pool::IRelationalService>( "POOL/Services/RelationalService" );
  if ( ! serviceHandle ) {
    throw cms::Exception( "cond::MetaData::MetaData: Could not retrieve the relational service" );
  }
  pool::IRelationalDomain& domain = serviceHandle->domainForConnection(m_con);
  m_session.reset( domain.newSession( m_con ) );
  if ( ! m_session->connect() ) {
    throw cms::Exception( "cond::MetaData::MetaData Could not connect to the database server." );
  }
}

cond::MetaData::~MetaData(){
  (*m_log)<<seal::Msg::Debug<< "Disconnecting..." << seal::flush;
  m_session->disconnect();
}

bool cond::MetaData::addMapping(const std::string& name, const std::string& iovtoken){
  (*m_log)<<seal::Msg::Debug<<"cond::MetaData::addMapping"<<seal::flush;
  if ( ! m_session->transaction().start() ) {
    throw cms::Exception( "cond::MetaData::addMapping Could not start transaction");
  }
  if(!m_session->userSchema().existsTable(cond::MetaDataNames::metadataTable())){
    this->createTable(cond::MetaDataNames::metadataTable());
  }else{
    m_table=&(m_session->userSchema().tableHandle(cond::MetaDataNames::metadataTable()));
  }
  pool::AttributeList data( m_table->description().columnNamesAndTypes() );
  pool::IRelationalTableDataEditor& dataEditor = m_table->dataEditor();
  data[cond::MetaDataNames::tagColumn()].setValue<std::string>(name);
  data[cond::MetaDataNames::tokenColumn()].setValue<std::string>(iovtoken);
  bool status= dataEditor.insertNewRow( data );
  if ( ! m_session->transaction().commit() ) {
    throw cms::Exception("cond::MetaData::addMapping Could not commit the transaction");
  }
  return status;
}

const std::string cond::MetaData::getToken( const std::string& name ){
  (*m_log)<<seal::Msg::Debug<<"cond::MetaData::getToken "<<name<<seal::flush;
  if(!m_table){
    m_table=&(m_session->userSchema().tableHandle( cond::MetaDataNames::metadataTable() ));
  }
  std::string iovtoken;
  if ( ! m_session->transaction().start() ) {
    throw cms::Exception( "cond::MetaData::getToken: Could not start a new transaction" );
  }
  std::auto_ptr< pool::IRelationalQuery > query( m_table->createQuery() );
  query->setRowCacheSize( 10 );
  pool::AttributeList emptyBindVariableList;
  std::string condition=cond::MetaDataNames::tagColumn()+"='"+name+"'";
  query->setCondition( condition, emptyBindVariableList );
  query->addToOutputList( cond::MetaDataNames::tokenColumn() );
  pool::IRelationalCursor& cursor = query->process();
  if ( cursor.start() ) {
    while( cursor.next() ) {
      const pool::AttributeList& row = cursor.currentRow();
      for ( pool::AttributeList::const_iterator iColumn = row.begin();
	    iColumn != row.end(); ++iColumn ) {
	//std::cout << iColumn->spec().name() << " : " << iColumn->getValueAsString() << "\t";
	iovtoken=iColumn->getValueAsString();
      }
      //std::cout << std::endl;
    }
  }
  if ( ! m_session->transaction().commit() ) {
    throw cms::Exception( "cond::MetaData::getToken: Could not commit a transaction" );
  }
  return iovtoken;
}
void cond::MetaData::createTable(const std::string& tabname){
  if ( ! m_session->transaction().start() ) {
    throw cms::Exception( "cond::MetaData::createTable Could not start transaction." );
  }
  pool::IRelationalSchema& schema=m_session->userSchema();
  seal::IHandle<pool::IRelationalService> serviceHandle=pool::POOLContext::context()->query<pool::IRelationalService>( "POOL/Services/RelationalService" );
  pool::IRelationalDomain& domain = serviceHandle->domainForConnection(m_con);
  std::auto_ptr< pool::IRelationalEditableTableDescription > desc( new pool::RelationalAccess::RelationalEditableTableDescription( *m_log, domain.flavorName() ) );
  desc->insertColumn(  cond::MetaDataNames::tagColumn(), pool::AttributeStaticTypeInfo<std::string>::type_name() );
  desc->insertColumn( cond::MetaDataNames::tokenColumn(), pool::AttributeStaticTypeInfo<std::string>::type_name() );
  std::vector<std::string> cols;
  cols.push_back( cond::MetaDataNames::tagColumn() );
  desc->setPrimaryKey(cols);
  desc->setNotNullConstraint( cond::MetaDataNames::tokenColumn() );
  m_table=&(schema.createTable(tabname,*desc));  
  m_table->privilegeManager().grantToPublic( pool::IRelationalTablePrivilegeManager::SELECT );
  if ( ! m_session->transaction().commit() ) {
    throw cms::Exception( "cond::MetaData::createTable: Could not commit a transaction" );
  }
}
