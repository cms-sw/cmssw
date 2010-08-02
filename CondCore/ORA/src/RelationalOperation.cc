#include "CondCore/ORA/interface/Exception.h"
#include "RelationalOperation.h"
// externals 
#include "CoralBase/Attribute.h"
#include "CoralBase/Blob.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IBulkOperation.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"

namespace ora {

  bool existAttribute(const std::string& attributeName,
                      const coral::AttributeList& data){
    bool found = false;
    for(coral::AttributeList::const_iterator iAttr = data.begin();
        iAttr!=data.end() && !found; ++iAttr){
      if( iAttr->specification().name() == attributeName ) found = true;
    }
    return found;
  }

  char* conditionOfType( ConditionType condType ){
    static char* cond[ 5 ] = { (char*)"=",(char*)">",(char*)"<",(char*)">=",(char*)"<=" };
    return cond[ condType ];
  }
  
}

ora::InputRelationalData::InputRelationalData():
  m_data(),
  m_setClause(""),
  m_whereClause(""){
}

ora::InputRelationalData::~InputRelationalData(){
}

void ora::InputRelationalData::addId(const std::string& columnName){
  if(!existAttribute( columnName, m_data )){
    m_data.extend<int>( columnName );
    if(!m_setClause.empty()) m_setClause += ", ";
    m_setClause += ( columnName +"= :"+columnName );
  }
}

void ora::InputRelationalData::addData(const std::string& columnName,
                                       const std::type_info& columnType ){
  if(!existAttribute( columnName, m_data )){
    m_data.extend( columnName, columnType );
    if(!m_setClause.empty()) m_setClause += ", ";
    m_setClause += ( columnName +"= :"+columnName );
  }
}

void ora::InputRelationalData::addBlobData(const std::string& columnName){
  if(!existAttribute( columnName, m_data )){
    m_data.extend<coral::Blob>( columnName );
    if(!m_setClause.empty()) m_setClause += ", ";
    m_setClause += ( columnName +"= :"+columnName );
  }
}

void ora::InputRelationalData::addWhereId( const std::string& columnName ){
  if(!existAttribute( columnName, m_data )){
    m_data.extend<int>( columnName );
  }
  if(!m_whereClause.empty()) m_whereClause += " AND ";
  m_whereClause += ( columnName +"= :"+columnName );
}

void ora::InputRelationalData::addWhereId( const std::string& columnName, ConditionType condType ){
  if(!existAttribute( columnName, m_data )){
    m_data.extend<int>( columnName );
    if(!m_whereClause.empty()) m_whereClause += " AND ";
    m_whereClause += ( columnName +conditionOfType(condType)+" :"+columnName );
  }
}

coral::AttributeList& ora::InputRelationalData::data(){
  return m_data;
}

coral::AttributeList& ora::InputRelationalData::whereData(){
  // data and where data are hosted in the same buffer (as for update operation)
  return m_data;
}

std::string& ora::InputRelationalData::updateClause(){
  return m_setClause;
}

std::string& ora::InputRelationalData::whereClause(){
  return m_whereClause;
}

ora::InsertOperation::InsertOperation( const std::string& tableName,
                                       coral::ISchema& schema ):
  InputRelationalData(),
  m_tableName( tableName ),
  m_schema( schema ){
}

ora::InsertOperation::~InsertOperation(){
}

bool
ora::InsertOperation::isRequired(){
  return false;
}

bool ora::InsertOperation::execute(){
  coral::ITable& table = m_schema.tableHandle( m_tableName );
  table.dataEditor().insertRow( data() );
  return true;
}

void ora::InsertOperation::reset(){
}

ora::BulkInsertOperation::BulkInsertOperation( const std::string& tableName,
                                               coral::ISchema& schema ):
  InputRelationalData(),
  m_tableName( tableName ),
  m_schema( schema ),
  m_bulkOperations(){
}

ora::BulkInsertOperation::~BulkInsertOperation(){
  for( std::vector<coral::IBulkOperation*>::iterator iB = m_bulkOperations.begin();
       iB != m_bulkOperations.end(); ++iB ){
    delete *iB;
  }
}

coral::IBulkOperation& ora::BulkInsertOperation::setUp( int rowCacheSize ){
  coral::ITable& table = m_schema.tableHandle( m_tableName );
  
  m_bulkOperations.push_back( table.dataEditor().bulkInsert( data(), rowCacheSize  ) );
  return *m_bulkOperations.back();
}

bool
ora::BulkInsertOperation::isRequired(){
  return false;
}

bool ora::BulkInsertOperation::execute(){
  for( std::vector<coral::IBulkOperation*>::iterator iB = m_bulkOperations.begin();
       iB != m_bulkOperations.end(); ++iB ){
    (*iB)->flush();
    delete *iB;
  }
  m_bulkOperations.clear();
  return true;
}

void ora::BulkInsertOperation::reset(){
  for( std::vector<coral::IBulkOperation*>::iterator iB = m_bulkOperations.begin();
       iB != m_bulkOperations.end(); ++iB ){
    delete *iB;
  }
  m_bulkOperations.clear();  
}

ora::UpdateOperation::UpdateOperation( const std::string& tableName,
                                       coral::ISchema& schema ):
  InputRelationalData(),
  m_tableName( tableName ),
  m_schema( schema ){
}

ora::UpdateOperation::~UpdateOperation(){
}

bool
ora::UpdateOperation::isRequired(){
  return true;
}

bool ora::UpdateOperation::execute(){
  bool ret = false;
  if( updateClause().size() && whereClause().size() ){
    coral::ITable& table = m_schema.tableHandle( m_tableName );
    long nr = table.dataEditor().updateRows( updateClause(), whereClause(), data() );
    ret = nr > 0;
  }
  return ret;
}

void ora::UpdateOperation::reset(){
}

ora::DeleteOperation::DeleteOperation( const std::string& tableName,
                                       coral::ISchema& schema ):
  InputRelationalData(),
  m_tableName( tableName ),
  m_schema( schema ){
}

ora::DeleteOperation::~DeleteOperation(){
}

bool
ora::DeleteOperation::isRequired(){
  return false;
}

bool ora::DeleteOperation::execute(){
  bool ret = false;
  if( whereClause().size() ){
    coral::ITable& table = m_schema.tableHandle( m_tableName );
    long nr = table.dataEditor().deleteRows( whereClause(), whereData() );
    ret = nr > 0;
  }
  return ret;
}

void ora::DeleteOperation::reset(){
}

ora::SelectOperation::SelectOperation( const std::string& tableName,
                                       coral::ISchema& schema ):
  m_spec( new coral::AttributeListSpecification ),
  m_whereData(),
  m_whereClause(""),
  m_orderByCols(),
  m_query(),
  m_cursor( 0 ),
  m_tableName( tableName ),
  m_schema( schema ){
}

ora::SelectOperation::~SelectOperation(){
  m_spec->release();
}

void ora::SelectOperation::addOrderId(const std::string& columnName){
  m_orderByCols.push_back( columnName );
}

bool ora::SelectOperation::nextCursorRow(){
  bool ret = false;
  if( m_query.get() ){
    ret = m_cursor->next();
    if(!ret) clear();
  }
  return ret;
}

void ora::SelectOperation::clear(){
  m_query.reset();
  m_cursor = 0;
}

void ora::SelectOperation::addId(const std::string& columnName){
  if(m_spec->index( columnName )==-1){
    m_spec->extend< int >( columnName );
  }
}

void ora::SelectOperation::addData(const std::string& columnName,
                                   const std::type_info& columnType ){
  if(m_spec->index( columnName )==-1){
    m_spec->extend( columnName, columnType );
  }
}

void ora::SelectOperation::addBlobData(const std::string& columnName ){
  if(m_spec->index( columnName )==-1){
    m_spec->extend<coral::Blob>( columnName );
  }
}

void ora::SelectOperation::addWhereId( const std::string& columnName ){
  if(!existAttribute( columnName, m_whereData )){
    m_whereData.extend<int>( columnName );
    if(!m_whereClause.empty()) m_whereClause += " AND ";
    m_whereClause += ( columnName +"= :"+columnName );
  }
}

coral::AttributeList& ora::SelectOperation::data(){
  if(!m_cursor) throwException( "Query on table "+m_tableName+" has not been executed.",
                                "ora::ReadOperation::data" );
  return const_cast<coral::AttributeList&>(m_cursor->currentRow());
}

coral::AttributeList& ora::SelectOperation::whereData(){
  return m_whereData;
}

std::string& ora::SelectOperation::whereClause(){
  return m_whereClause;
}

void ora::SelectOperation::execute(){
  m_cursor = 0;
  coral::ITable& table = m_schema.tableHandle( m_tableName );
  m_query.reset( table.newQuery() );
  for ( coral::AttributeListSpecification::const_iterator iSpec = m_spec->begin();
        iSpec != m_spec->end(); ++iSpec ) {
    m_query->addToOutputList( iSpec->name() );
    m_query->defineOutputType( iSpec->name(),iSpec->typeName());
  }
  for(std::vector<std::string>::iterator iCol = m_orderByCols.begin();
      iCol != m_orderByCols.end(); iCol++ ){
    m_query->addToOrderList( *iCol );
  }
  m_query->setCondition( m_whereClause, m_whereData );
  m_query->setRowCacheSize( 100 ); // We should better define this value !!!
  m_cursor = &m_query->execute();
}

coral::AttributeListSpecification& ora::SelectOperation::attributeListSpecification(){
  return *m_spec;
}

