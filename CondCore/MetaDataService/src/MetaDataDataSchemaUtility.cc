#include "CondCore/MetaDataService/interface/MetaDataSchemaUtility.h"
#include "CondCore/MetaDataService/interface/MetaDataNames.h"
#include "CondCore/MetaDataService/interface/MetaDataExceptions.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
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
cond::MetaDataSchemaUtility::MetaDataSchemaUtility(const CoralTransaction& coraldb){
}
void
cond::MetaDataSchemaUtility::create(){
 coral::ISchema& schema=m_coraldb.nominalSchema();
  coral::TableDescription description;
  description.setName( tabname );
  description.insertColumn(  cond::MetaDataNames::tagColumn(), coral::AttributeSpecification::typeNameForId( typeid(std::string)) );
  description.insertColumn( cond::MetaDataNames::tokenColumn(), coral::AttributeSpecification::typeNameForId( typeid(std::string)) );
  description.insertColumn( cond::MetaDataNames::timetypeColumn(), coral::AttributeSpecification::typeNameForId( typeid(int)) );
  std::vector<std::string> cols;
  cols.push_back( cond::MetaDataNames::tagColumn() );
  description.setPrimaryKey(cols);
  description.setNotNullConstraint( cond::MetaDataNames::tokenColumn() );
  coral::ITable& table=schema.createTable(description);
  table.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select);
}
void
cond::MetaDataSchemaUtility::drop(){
  
}
