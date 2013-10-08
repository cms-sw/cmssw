#include "CondCore/CondDB/interface/Exception.h"
#include "SessionImpl.h"
#include "GTSchema.h"
//
namespace conddb {

bool GLOBAL_TAG::exists( SessionImpl& session ){
  return existsTable( session.coralSchema(), tname );
}

bool GLOBAL_TAG::select( const std::string& name, SessionImpl& session ){
  Query< NAME > q( session.coralSchema() );
  q.addCondition<NAME>( name );
  for ( auto row : q ) {}

  return q.retrievedRows();
}

bool GLOBAL_TAG::select( const std::string& name, conddb::Time_t& validity, boost::posix_time::ptime& snapshotTime, SessionImpl& session ){
  Query< VALIDITY, SNAPSHOT_TIME > q( session.coralSchema() );
  q.addCondition<NAME>( name );
  for ( auto row : q ) std::tie( validity, snapshotTime ) = row;

  return q.retrievedRows();
}
    
bool GLOBAL_TAG::select( const std::string& name, conddb::Time_t& validity, std::string& description, std::string& release, 
			 boost::posix_time::ptime& snapshotTime, SessionImpl& session ){
  Query< VALIDITY, DESCRIPTION, RELEASE, SNAPSHOT_TIME > q( session.coralSchema() );
  q.addCondition<NAME>( name );
  for ( auto row : q ) std::tie( validity, description, release, snapshotTime ) = row;
  return q.retrievedRows();
}
    
void GLOBAL_TAG::insert( const std::string& name, conddb::Time_t validity, std::string& description, const std::string& release, 
			 const boost::posix_time::ptime& snapshotTime, const boost::posix_time::ptime& insertionTime, SessionImpl& session ){
  RowBuffer< NAME, VALIDITY, DESCRIPTION, RELEASE, SNAPSHOT_TIME, INSERTION_TIME > 
    dataToInsert( std::tie( name, validity, description, release, snapshotTime, insertionTime ) );
  insertInTable( session.coralSchema(), tname, dataToInsert.get() );
}

void GLOBAL_TAG::update( const std::string& name, conddb::Time_t validity, std::string& description, const std::string& release, 
			 const boost::posix_time::ptime& snapshotTime, const boost::posix_time::ptime& insertionTime, SessionImpl& session ){
  UpdateBuffer buffer;
  buffer.setColumnData< VALIDITY, DESCRIPTION, RELEASE, SNAPSHOT_TIME, INSERTION_TIME >( std::tie( validity, description, release, snapshotTime, insertionTime  ) );
  buffer.addWhereCondition<NAME>( name );
  updateTable( session.coralSchema(), tname, buffer );  
}

bool GLOBAL_TAG_MAP::exists( SessionImpl& session ){
  return existsTable( session.coralSchema(), tname );
}
    
bool GLOBAL_TAG_MAP::select( const std::string& gtName, std::vector<std::tuple<std::string,std::string,std::string> >& tags, SessionImpl& session ){
  Query< RECORD, LABEL, TAG_NAME > q( session.coralSchema() );
  q.addCondition< GLOBAL_TAG_NAME >( gtName );
  q.addOrderClause<RECORD>();
  q.addOrderClause<LABEL>();
  for ( auto row : q ) {
    tags.push_back( row );
  }
  return q.retrievedRows();
}
    
void GLOBAL_TAG_MAP::insert( const std::string& gtName, const std::vector<std::tuple<std::string,std::string,std::string> >& tags, SessionImpl& session ){
  BulkInserter<GLOBAL_TAG_NAME, RECORD, LABEL, TAG_NAME > inserter( session.coralSchema(), tname );
  for( auto row : tags ) inserter.insert( std::tuple_cat( std::tie( gtName ),row ) );
  inserter.flush();  
}
  
bool gtDb::exists( SessionImpl& session ){
  size_t ntables = 0;
  if( GLOBAL_TAG::exists( session ) ) ntables++;
  if( GLOBAL_TAG_MAP::exists( session ) ) ntables++;
  if( ntables && ntables<2 ) conddb::throwException( "The GT Database is incomplete or corrupted.","gtDb::exists" );
  return ntables;
  
}

}
