#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/MetaDataService/interface/MetaDataNames.h"
#include "CondCore/MetaDataService/interface/MetaDataExceptions.h"
#include "CondCore/DBCommon/interface/ServiceLoader.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "RelationalAccess/AccessMode.h"
#include "RelationalAccess/IRelationalService.h"
#include "RelationalAccess/RelationalServiceException.h"
#include "RelationalAccess/IRelationalDomain.h"
#include "RelationalAccess/SchemaException.h"
#include "RelationalAccess/ISession.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/IPrimaryKey.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "CoralBase/Exception.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"
cond::MetaData::MetaData(const std::string& connectionString, cond::ServiceLoader& loader):
  m_con(connectionString),m_loader(loader){
  if(!m_loader.hasMessageService()){
    m_loader.loadMessageService();
  }
  if( !m_loader.hasAuthenticationService() ){
    m_loader.loadAuthenticationService();
  }
  m_service=&(m_loader.loadRelationalService());
  m_session=0;
  m_mode=cond::ReadWriteCreate;
}
cond::MetaData::~MetaData(){
}
void cond::MetaData::connect( cond::ConnectMode mod ){
  coral::IRelationalDomain& domain = m_service->domainForConnection(m_con);
  m_session=domain.newSession( m_con ) ;
  m_mode=mod;
  try{
    if( m_mode == cond::ReadOnly){
      m_session->connect(coral::ReadOnly);
    }else{
      m_session->connect(coral::Update);
    }
    m_session->startUserSession();
  }catch(std::exception& er){
    throw cond::Exception("MetaData::MetaData connect")<<er.what();
  }catch(...){
    throw cond::Exception( "MetaData::connect caught unknown exception" );
  }
}
void cond::MetaData::disconnect(){
  m_session->disconnect();
  delete m_session;
  m_session=0;
}
bool cond::MetaData::addMapping(const std::string& name, const std::string& iovtoken){
  try{
    m_session->transaction().start(false);
    if(!m_session->nominalSchema().existsTable(cond::MetaDataNames::metadataTable())){
      this->createTable(cond::MetaDataNames::metadataTable());
    }
    coral::ITable& mytable=m_session->nominalSchema().tableHandle(cond::MetaDataNames::metadataTable());
    coral::AttributeList rowBuffer;
    coral::ITableDataEditor& dataEditor = mytable.dataEditor();
    dataEditor.rowBuffer( rowBuffer );
    rowBuffer[cond::MetaDataNames::tagColumn()].data<std::string>()=name;
    rowBuffer[cond::MetaDataNames::tokenColumn()].data<std::string>()=iovtoken;
    dataEditor.insertRow( rowBuffer );
    m_session->transaction().commit() ;
  }catch( coral::DuplicateEntryInUniqueKeyException& er ){
    m_session->transaction().rollback() ;
    throw cond::MetaDataDuplicateEntryException("addMapping",name);
  }catch(std::exception& er){
    m_session->transaction().rollback() ;
    throw cond::Exception(er.what());
  }catch(...){
    m_session->transaction().rollback() ;
    throw cond::Exception( "MetaData::addMapping Could not commit the transaction" );
  }
  return true;
}
bool cond::MetaData::replaceToken(const std::string& name, const std::string& newtoken){
  try{
    m_session->transaction().start(false);
    if(!m_session->nominalSchema().existsTable(cond::MetaDataNames::metadataTable())){
      throw cond::Exception( "MetaData::replaceToken MetaData table doesnot exist" );
    }
    coral::ITable& mytable=m_session->nominalSchema().tableHandle(cond::MetaDataNames::metadataTable());
    coral::AttributeList inputData;
    coral::ITableDataEditor& dataEditor = mytable.dataEditor();
    inputData.extend<std::string>("newToken");
    inputData.extend<std::string>("oldTag");
    inputData[0].data<std::string>() = newtoken;
    inputData[1].data<std::string>() = name;
    std::string setClause(cond::MetaDataNames::tokenColumn());
    setClause+="= :newToken";
    std::string condition( cond::MetaDataNames::tagColumn() );
    condition+="= :oldTag";
    dataEditor.updateRows( setClause, condition, inputData );
    m_session->transaction().commit() ;
  }catch( coral::DuplicateEntryInUniqueKeyException& er ){
    m_session->transaction().rollback();
    throw cond::MetaDataDuplicateEntryException("MetaData::replaceToken",name);
  }catch(std::exception& er){
    m_session->transaction().rollback() ;
    throw cond::Exception(er.what());
  }catch(...){
    m_session->transaction().rollback() ;
    throw cond::Exception( "MetaData::replaceToken Could not commit the transaction" );
  }
  return true;
}
const std::string cond::MetaData::getToken( const std::string& name ){
  std::string iovtoken;
  try{
    if( m_mode!=cond::ReadOnly ){
      m_session->transaction().start(false);
    }else{
      m_session->transaction().start(true);
    }
    coral::ITable& mytable=m_session->nominalSchema().tableHandle( cond::MetaDataNames::metadataTable() );
    std::auto_ptr< coral::IQuery > query(mytable.newQuery());
    query->setRowCacheSize( 100 );
    coral::AttributeList emptyBindVariableList;
    std::string condition=cond::MetaDataNames::tagColumn()+" = '"+name+"'";
    query->setCondition( condition, emptyBindVariableList );
    query->addToOutputList( cond::MetaDataNames::tokenColumn() );
    coral::ICursor& cursor = query->execute();
    while( cursor.next() ) {
      const coral::AttributeList& row = cursor.currentRow();
      iovtoken=row[ cond::MetaDataNames::tokenColumn() ].data<std::string>();
    }
    m_session->transaction().commit();
  }catch(const coral::TableNotExistingException& er){
    m_session->transaction().commit();
    return "";
  }catch(const std::exception& er){
    m_session->transaction().rollback() ;
    throw cond::Exception( std::string("MetaData::getToken error: ")+er.what() );
  }catch(...){
    m_session->transaction().rollback() ;
    throw cond::Exception( "MetaData::getToken: unknow exception" );
  }
  return iovtoken;
}
void cond::MetaData::createTable(const std::string& tabname){
  coral::ISchema& schema=m_session->nominalSchema();
  coral::TableDescription description;
  description.setName( tabname );
  description.insertColumn(  cond::MetaDataNames::tagColumn(), coral::AttributeSpecification::typeNameForId( typeid(std::string)) );
  description.insertColumn( cond::MetaDataNames::tokenColumn(), coral::AttributeSpecification::typeNameForId( typeid(std::string)) );
  std::vector<std::string> cols;
  cols.push_back( cond::MetaDataNames::tagColumn() );
  description.setPrimaryKey(cols);
  description.setNotNullConstraint( cond::MetaDataNames::tokenColumn() );
  coral::ITable& table=schema.createTable(description);
  table.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select);
}
bool cond::MetaData::hasTag( const std::string& name ) const{
  bool result=false;
  try{
    if( m_mode!=cond::ReadOnly ){
      m_session->transaction().start(false);
    }else{
      m_session->transaction().start(true);
    }
    coral::ITable& mytable=m_session->nominalSchema().tableHandle( cond::MetaDataNames::metadataTable() );
    std::auto_ptr< coral::IQuery > query(mytable.newQuery());
    coral::AttributeList emptyBindVariableList;
    std::string condition=cond::MetaDataNames::tagColumn()+" = '"+name+"'";
    query->setCondition( condition, emptyBindVariableList );
    coral::ICursor& cursor = query->execute();
    if( cursor.next() ) result=true;
    cursor.close();
    m_session->transaction().commit();
  }catch(const coral::TableNotExistingException& er){
    m_session->transaction().commit();
    return false;
  }catch(const std::exception& er){
    m_session->transaction().rollback() ;
    throw cond::Exception( std::string("MetaData::hasTag: " )+er.what() );
  }catch(...){
    m_session->transaction().rollback() ;
    throw cond::Exception( "MetaData::hasTag: unknown exception ");
  }
  return result;
}
void cond::MetaData::listAllTags( std::vector<std::string>& result ) const{
  try{
    if( m_mode!=cond::ReadOnly ){
      m_session->transaction().start(false);
    }else{
      m_session->transaction().start(true);
    }
    coral::ITable& mytable=m_session->nominalSchema().tableHandle( cond::MetaDataNames::metadataTable() );
    std::auto_ptr< coral::IQuery > query(mytable.newQuery());
    query->addToOutputList( cond::MetaDataNames::tagColumn() );
    query->setMemoryCacheSize( 100 );
    coral::ICursor& cursor = query->execute();
    while( cursor.next() ){
      const coral::AttributeList& row = cursor.currentRow();
      result.push_back(row[cond::MetaDataNames::tagColumn()].data<std::string>());
    }
    cursor.close();
    m_session->transaction().commit();
  }catch(const coral::TableNotExistingException& er){
    m_session->transaction().commit();
    return;
  }catch(const std::exception& er){
    m_session->transaction().rollback() ;
    throw cond::Exception( std::string("MetaData::listAllTag: " )+er.what() );
  }catch(...){
    m_session->transaction().rollback() ;
    throw cond::Exception( "MetaData::listAllTag: unknown exception ");
  }
}
