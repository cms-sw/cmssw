#include "IDatabaseSchema.h"
#include "OraDatabaseSchema.h"
#include "PoolDatabaseSchema.h"

std::string ora::IMainTable::schemaVersionParameterName(){
  static std::string s_name("SCHEMA_VERSION");
  return s_name;
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

