#include "CondCore/ORA/interface/Exception.h"
#include "MultiRecordInsertOperation.h"
#include "RecordManip.h"
// externals 
#include "CoralBase/Blob.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IBulkOperation.h"

ora::InsertCache::InsertCache( const RecordSpec& spec, 
                               const coral::AttributeList& data ):
  m_spec(spec),m_records(),m_data( data ){
}

ora::InsertCache::~InsertCache(){
  for(std::vector<Record*>::const_iterator iR = m_records.begin();
      iR != m_records.end(); ++iR ){
    delete *iR;
  }
}

void ora::InsertCache::processNextIteration(){
  Record* rec = new Record(m_spec);
  newRecordFromAttributeList(*rec, m_data );
  m_records.push_back( rec );
}

const std::vector<ora::Record*>& ora::InsertCache::records() const {
  return m_records;
}

ora::MultiRecordInsertOperation::MultiRecordInsertOperation( const std::string& tableName, 
                                                             coral::ISchema& schema ):
  m_relationalData(),m_tableName( tableName ),m_schema( schema ),m_bulkInserts(){
}

ora::MultiRecordInsertOperation::~MultiRecordInsertOperation(){
  for( std::vector<InsertCache*>::iterator iB = m_bulkInserts.begin();
       iB != m_bulkInserts.end(); ++iB ){
    delete *iB;
  }
}

ora::InsertCache& ora::MultiRecordInsertOperation::setUp( int ){
  m_bulkInserts.push_back( new InsertCache( m_spec, m_relationalData.data() ) );
  return *m_bulkInserts.back(); 
}

void ora::MultiRecordInsertOperation::addId( const std::string& columnName ){
  m_relationalData.addId( columnName );
  m_spec.add( columnName, typeid(int) );
}

void ora::MultiRecordInsertOperation::addData( const std::string& columnName, 
                                         const std::type_info& columnType ){
  m_relationalData.addData( columnName, columnType );
  m_spec.add( columnName, columnType );
}

void ora::MultiRecordInsertOperation::addBlobData(const std::string& columnName){
  m_relationalData.addBlobData( columnName );
  m_spec.add( columnName, typeid(coral::Blob) );
}
    
void ora::MultiRecordInsertOperation::addWhereId( const std::string& columnName ){
  m_relationalData.addWhereId( columnName );
}

coral::AttributeList& ora::MultiRecordInsertOperation::data(){
  return m_relationalData.data();
}

coral::AttributeList& ora::MultiRecordInsertOperation::whereData(){
  return m_relationalData.whereData();
}

std::string& ora::MultiRecordInsertOperation::whereClause(){
  return m_relationalData.whereClause();
}

bool
ora::MultiRecordInsertOperation::isRequired(){
  return false;
}

bool ora::MultiRecordInsertOperation::execute(){
  for( std::vector<InsertCache*>::iterator iB = m_bulkInserts.begin();
       iB != m_bulkInserts.end(); ++iB ){
    coral::ITable& table = m_schema.tableHandle( m_tableName );
    std::auto_ptr<coral::IBulkOperation> bulkExecute( table.dataEditor().bulkInsert( m_relationalData.data(), (*iB)->records().size()  ) );
    unsigned int i=0;
    for( std::vector<Record*>::const_iterator iR = (*iB)->records().begin();
         iR != (*iB)->records().end(); ++iR ){
      i++;
      newAttributeListFromRecord( m_relationalData.data(), *(*iR) );
      bulkExecute->processNextIteration();
      if( i== INSERTCACHESIZE ){
         bulkExecute->flush();
         i = 0;
      }
    }
    bulkExecute->flush();
    delete *iB;
  }
  m_bulkInserts.clear();
  return true;
}

void ora::MultiRecordInsertOperation::reset(){
  for( std::vector<InsertCache*>::iterator iB = m_bulkInserts.begin();
       iB != m_bulkInserts.end(); ++iB ){
    delete *iB;
  }
  m_bulkInserts.clear();  
}
  
