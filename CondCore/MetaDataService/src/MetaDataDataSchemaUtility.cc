#include "CondCore/MetaDataService/interface/MetaDataSchemaUtility.h"
#include "CondCore/MetaDataService/interface/MetaDataNames.h"
#include "RelationalAccess/SchemaException.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"
cond::MetaDataSchemaUtility::MetaDataSchemaUtility(cond::DbSession& coraldb):m_coraldb(coraldb){
}
cond::MetaDataSchemaUtility::~MetaDataSchemaUtility(){}

void
cond::MetaDataSchemaUtility::create(){
  try{
    coral::ISchema& schema=m_coraldb.nominalSchema();
    coral::TableDescription description;
    description.setName( cond::MetaDataNames::metadataTable() );
    description.insertColumn(  cond::MetaDataNames::tagColumn(), coral::AttributeSpecification::typeNameForId( typeid(std::string)) );
    description.insertColumn( cond::MetaDataNames::tokenColumn(), coral::AttributeSpecification::typeNameForId( typeid(std::string)) );
    description.insertColumn( cond::MetaDataNames::timetypeColumn(), coral::AttributeSpecification::typeNameForId( typeid(int)) );
    std::vector<std::string> cols;
    cols.push_back( cond::MetaDataNames::tagColumn() );
    description.setPrimaryKey(cols);
    description.setNotNullConstraint( cond::MetaDataNames::tokenColumn() );
    coral::ITable& table=schema.createTable(description);
    table.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select);
  }catch( const coral::TableAlreadyExistingException& er ){
    //must catch and ignore this exception!!
    //std::cout<<"table alreay existing, not creating a new one"<<std::endl;
  }
}
void
cond::MetaDataSchemaUtility::drop(){
  coral::ISchema& schema=m_coraldb.nominalSchema();
  try{
    schema.dropTable(cond::MetaDataNames::metadataTable());
  }catch(coral::TableNotExistingException& er){
    //must catch and ignore this exception!!
    //ok do nothing
  }
}
