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
lumi::idDealer::idDealer( coral::ISchema& schema):m_schema(schema),m_idtablecolumnName(lumi::LumiNames::idTableColumnName()),m_idtablecolumnType(lumi::LumiNames::idTableColumnType()){
}
unsigned int lumi::idDealer::getIDforTable( const std::string& tableName ){
  std::string idtableName=lumi::LumiNames::idTableName(tableName);
  coral::IQuery* q=m_schema.tableHandle(idtableName).newQuery();
  q->addToOutputList(m_idtablecolumnName);
  q->setForUpdate(); //lock it
  coral::ICursor& cursor=q->execute();
  unsigned int result=0;
  while ( cursor.next() ){
    const coral::AttributeList& row = cursor.currentRow();
    result = row[m_idtablecolumnName].data<unsigned int>();
  }
  cursor.close();
  delete q;
  return result;
}
unsigned int lumi::idDealer::generateNextIDForTable( const std::string& tableName ){
  std::string idtableName=lumi::LumiNames::idTableName(tableName);
  coral::IQuery* q=m_schema.tableHandle(idtableName).newQuery();
  q->addToOutputList(m_idtablecolumnName);
  q->setForUpdate(); //lock it
  coral::ICursor& cursor=q->execute();
  unsigned int result=0;
  while ( cursor.next() ){
    const coral::AttributeList& row = cursor.currentRow();
    result = row[m_idtablecolumnName].data<unsigned int>();
  }
  coral::ITableDataEditor& dataEditor=m_schema.tableHandle(idtableName).dataEditor();
  coral::AttributeList inputData;
  if( result==0 ){
    dataEditor.rowBuffer(inputData);
    inputData[m_idtablecolumnName].data<unsigned int>()=result+1;
    dataEditor.insertRow(inputData);
  }else{
    inputData.extend("newid",m_idtablecolumnType);
    inputData["newid"].data<unsigned int>()=result+1;
    dataEditor.updateRows(m_idtablecolumnName+"= :newid","",inputData);
  }
  delete q;
  return result+1;
}
