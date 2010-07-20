#include "CondCore/ORA/interface/Exception.h"
#include "MultiRecordSelectOperation.h"
// externals 
#include "CoralBase/Attribute.h"

ora::MultiRecordSelectOperation::MultiRecordSelectOperation( const std::string& tableName,
                                                              coral::ISchema& schema ):
  m_query( tableName, schema ),
  m_idCols(),
  m_cache(),
  m_row(){
  //m_row( 0 ){
}

ora::MultiRecordSelectOperation::~MultiRecordSelectOperation(){
}

void ora::MultiRecordSelectOperation::addOrderId(const std::string& columnName){
  m_query.addOrderId( columnName );
  m_idCols.push_back( columnName );
}

void ora::MultiRecordSelectOperation::selectRow( const std::vector<int>& selection ){
  //m_row = &m_cache.lookup( selection );
  m_row = m_cache.lookupAndClear( selection );
}

size_t ora::MultiRecordSelectOperation::selectionSize( const std::vector<int>& selection,
                                                        size_t numberOfIndexes ){
  if(m_cache.size()==0) return 0;
  return m_cache.branchSize(selection, numberOfIndexes);
}

void ora::MultiRecordSelectOperation::clear(){
  m_row.reset();
  //m_row = 0;
  m_cache.clear();
  //m_idCols.clear();
  m_query.clear();
}

void ora::MultiRecordSelectOperation::addId(const std::string& columnName){
  m_query.addId( columnName );
}

void ora::MultiRecordSelectOperation::addData(const std::string& columnName,
                                               const std::type_info& columnType ){
  m_query.addData( columnName, columnType );  
}

void ora::MultiRecordSelectOperation::addBlobData(const std::string& columnName){
  m_query.addBlobData( columnName );  
}

void ora::MultiRecordSelectOperation::addWhereId(const std::string& columnName){
  m_query.addWhereId( columnName );
  
}

coral::AttributeList& ora::MultiRecordSelectOperation::data(){
  //if(!m_row){
  if(!m_row.get()){
    throwException( "No record available.",
                    "MultiRecordSelectOperation::data" );
  }
  return *m_row;
}

coral::AttributeList& ora::MultiRecordSelectOperation::whereData(){
  return m_query.whereData();
}

std::string& ora::MultiRecordSelectOperation::whereClause(){
  return m_query.whereClause();
}

void ora::MultiRecordSelectOperation::execute(){
  //m_row = 0;
  m_row.reset();
  m_cache.clear();
  m_query.execute();
  while( m_query.nextCursorRow() ){
    std::vector<int> indexes;
    coral::AttributeList& row = m_query.data();
    for(size_t i=0;i<m_idCols.size();i++){
      indexes.push_back( row[m_idCols[i]].data<int>() );
    }
    m_cache.push( indexes,row );
  }
  m_query.clear();
}


