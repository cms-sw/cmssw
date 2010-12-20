#include "CondCore/ORA/interface/Exception.h"
#include "MultiRecordSelectOperation.h"
#include "RecordManip.h"
// externals 
#include "CoralBase/Blob.h"

ora::MultiRecordSelectOperation::MultiRecordSelectOperation( const std::string& tableName,
                                                              coral::ISchema& schema ):
  m_query( tableName, schema ),
  m_idCols(),
  m_cache(),
  m_spec(),
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
  if(!m_row.get())
    throwException( "No row available.",
                    "MultiRecordSelectOperation::selectRow" );
  //m_row = &m_cache.lookup( selection );
  Record rec; m_cache.lookupAndClear( selection,rec );
  newAttributeListFromRecord( *m_row, rec );
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
  m_spec.add( columnName, typeid(int) );
}

void ora::MultiRecordSelectOperation::addData(const std::string& columnName,
                                               const std::type_info& columnType ){
  m_query.addData( columnName, columnType );  
  m_spec.add( columnName, columnType );
}

void ora::MultiRecordSelectOperation::addBlobData(const std::string& columnName){
  m_query.addBlobData( columnName );  
  m_spec.add( columnName, typeid(coral::Blob) );
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
  //  m_row.reset();
  m_cache.clear();
  m_query.execute();
  while( m_query.nextCursorRow() ){
    std::vector<int> indexes;
    coral::AttributeList& row = m_query.data();
    for(size_t i=0;i<m_idCols.size();i++){
      indexes.push_back( row[m_idCols[i]].data<int>() );
    }
    Record rec(m_spec);
    newRecordFromAttributeList(rec, row );
    m_cache.push( indexes, rec );
  }
  m_row.reset(new coral::AttributeList(m_query.attributeListSpecification(), true ));
  m_query.clear();
}


