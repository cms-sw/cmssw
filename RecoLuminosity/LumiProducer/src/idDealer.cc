#include "RecoLuminosity/LumiProducer/interface/idDealer.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
//#include <iostream>
lumi::idDealer::idDealer( coral::ISchema& schema):m_schema(schema),m_idtablecolumnName(lumi::LumiNames::idTableColumnName()),m_idtablecolumnType(lumi::LumiNames::idTableColumnType()){
}
unsigned long long lumi::idDealer::getIDforTable( const std::string& tableName ){
  std::string idtableName=lumi::LumiNames::idTableName(tableName);
  coral::IQuery* q=m_schema.tableHandle(idtableName).newQuery();
  q->addToOutputList(m_idtablecolumnName);
  q->setForUpdate(); //lock it
  coral::ICursor& cursor=q->execute();
  unsigned long long result=0;
  while ( cursor.next() ){
    const coral::AttributeList& row = cursor.currentRow();
    result = row[m_idtablecolumnName].data<unsigned long long>();
  }
  cursor.close();
  delete q;
  return result;
}
unsigned long long lumi::idDealer::generateNextIDForTable( const std::string& tableName ){
  std::string idtableName=lumi::LumiNames::idTableName(tableName);
  coral::IQuery* q=m_schema.tableHandle(idtableName).newQuery();
  q->addToOutputList(m_idtablecolumnName);
  q->setForUpdate(); //lock it
  coral::ICursor& cursor=q->execute();
  unsigned long long result=0;
  while ( cursor.next() ){
    const coral::AttributeList& row = cursor.currentRow();
    result = row[m_idtablecolumnName].data<unsigned long long>();
  }
  coral::ITableDataEditor& dataEditor=m_schema.tableHandle(idtableName).dataEditor();
  coral::AttributeList inputData;
  dataEditor.updateRows(m_idtablecolumnName+"="+m_idtablecolumnName+"+1","",inputData);
  delete q;
  return result+1;
}
