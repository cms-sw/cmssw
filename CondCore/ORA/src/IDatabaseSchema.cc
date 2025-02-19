#include "IDatabaseSchema.h"
#include "OraDatabaseSchema.h"
#include "PoolDatabaseSchema.h"
// externals
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITablePrivilegeManager.h"

std::string ora::poolSchemaVersion(){
  static std::string s_version("POOL");
  return s_version;
}

void ora::setTableAccessPermission( coral::ITable& table, 
				    const std::string& principal, 
				    bool forWrite ){
  table.privilegeManager().grantToUser( principal, coral::ITablePrivilegeManager::Select );
  if(forWrite){
    table.privilegeManager().grantToUser( principal, coral::ITablePrivilegeManager::Update );
    table.privilegeManager().grantToUser( principal, coral::ITablePrivilegeManager::Insert );
    table.privilegeManager().grantToUser( principal, coral::ITablePrivilegeManager::Delete );
  } 
}

ora::IDatabaseTable::IDatabaseTable( coral::ISchema& schema ):
  m_schema( schema ){
}

coral::ISchema& ora::IDatabaseTable::schema(){
  return m_schema;
}

void ora::IDatabaseTable::setAccessPermission( const std::string& principal, 
					       bool forWrite ){
  coral::ITable& coralHandle = m_schema.tableHandle( name() );
  setTableAccessPermission( coralHandle, principal, forWrite );
}

std::string ora::IMainTable::versionParameterName(){
  static std::string s_name("SCHEMA_VERSION");
  return s_name;
}

std::string ora::IMainTable::userSchemaVersionParameterName(){
  static std::string s_name("USER_SCHEMA_VERSION");
  return s_name;
}

ora::IMainTable::IMainTable( coral::ISchema& schema ):
  IDatabaseTable( schema ){
}

ora::ISequenceTable::ISequenceTable( coral::ISchema& schema ):
  IDatabaseTable( schema ){
}

ora::MappingRawElement::MappingRawElement():
  scopeName(""),
  variableName(""),
  variableType(""),
  elementType(""),
  tableName(""),
  columns(){
}

ora::MappingRawElement::MappingRawElement(const MappingRawElement& rhs):
  scopeName( rhs.scopeName ),
  variableName( rhs.variableName ),
  variableType( rhs.variableType ),
  elementType( rhs.elementType ),
  tableName( rhs.tableName ),
  columns( rhs.columns ){
}

ora::MappingRawElement& ora::MappingRawElement::operator==(const MappingRawElement& rhs){
  scopeName = rhs.scopeName;
  variableName = rhs.variableName;
  variableType = rhs.variableType;
  elementType = rhs.elementType;
  tableName = rhs.tableName;
  columns = rhs.columns;
  return *this;
}

std::string ora::MappingRawElement::emptyScope(){
  static std::string s_scope("[]");
  return s_scope;
}

ora::MappingRawData::MappingRawData():  
  version( "" ),
  elements(){
}

ora::MappingRawData::MappingRawData( const std::string& vers ):  
  version( vers ),
  elements(){
}

ora::MappingRawElement& ora::MappingRawData::addElement( int elementId ){
  std::map< int, MappingRawElement>::iterator iElem = elements.find( elementId );
  if( iElem == elements.end() ){
    iElem = elements.insert( std::make_pair( elementId, MappingRawElement() ) ).first;
  }
  return iElem->second;
}

ora::ContainerHeaderData::ContainerHeaderData():
  id(-1),
  className(""),
  numberOfObjects(0){
}

ora::ContainerHeaderData::ContainerHeaderData( int contId,
                                               const std::string& classN,
                                               unsigned int numberObj ):
  id(contId),
  className(classN),
  numberOfObjects(numberObj){
}

ora::ContainerHeaderData::ContainerHeaderData( const ContainerHeaderData& rhs ):
  id(rhs.id),
  className(rhs.className),
  numberOfObjects(rhs.numberOfObjects){
}

ora::ContainerHeaderData& ora::ContainerHeaderData::operator=( const ContainerHeaderData& rhs ){
  id = rhs.id;
  className = rhs.className;
  numberOfObjects = rhs.numberOfObjects;
  return *this;
}

ora::IContainerHeaderTable::IContainerHeaderTable( coral::ISchema& schema ):
  IDatabaseTable( schema ){
}

ora::INamingServiceTable::INamingServiceTable( coral::ISchema& schema ):
  IDatabaseTable( schema ){
}

ora::IDatabaseSchema* ora::IDatabaseSchema::createSchemaHandle( coral::ISchema& schema ){
  IDatabaseSchema* dbSchema = 0;
  if( !OraDatabaseSchema::existsMainTable( schema ) ){
    if( PoolDatabaseSchema::existsMainTable( schema ) ) dbSchema = new PoolDatabaseSchema( schema );
  }
  if( ! dbSchema ) dbSchema = new OraDatabaseSchema( schema );
  return dbSchema;
}

ora::IDatabaseSchema::IDatabaseSchema( coral::ISchema& schema ):
  m_schema( schema ){
}

coral::ISchema& ora::IDatabaseSchema::storageSchema(){
  return m_schema;
}

