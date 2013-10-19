#include "CondCore/CondDB/interface/Exception.h"
#include "IOVSchema.h"
#include "SessionImpl.h"
//

namespace cond {

  namespace persistency {

    bool TAG::exists( SessionImpl& session ){
      return existsTable( session.coralSchema(), tname );
    }
   
    void TAG::create( SessionImpl& session ){
      if( exists( session ) ){
	throwException( "TAG table already exists in this schema.",
			"TAG::create");
      }
      TableDescription< NAME, TIME_TYPE, OBJECT_TYPE, SYNCHRONIZATION, END_OF_VALIDITY, DESCRIPTION, LAST_VALIDATED_TIME, INSERTION_TIME, MODIFICATION_TIME > descr( tname );
      descr.setPrimaryKey<NAME>();
      createTable( session.coralSchema(), descr.get() );
    }
    
    bool TAG::select( const std::string& name, 
		      SessionImpl& session ){
      Query< NAME > q( session.coralSchema() );
      q.addCondition<NAME>( name );
      for ( auto row : q ) {}
      
      return q.retrievedRows();
    }
    
    bool TAG::select( const std::string& name, 
		      cond::TimeType& timeType, 
		      std::string& objectType, 
		      cond::Time_t& endOfValidity,
		      std::string& description, 
		      cond::Time_t&  lastValidatedTime, 
		      SessionImpl& session ){
      Query< TIME_TYPE, OBJECT_TYPE, END_OF_VALIDITY, DESCRIPTION, LAST_VALIDATED_TIME > q( session.coralSchema() );
      q.addCondition<NAME>( name );
      for ( auto row : q ) std::tie( timeType, objectType, endOfValidity, description, lastValidatedTime ) = row;
      
      return q.retrievedRows();
    }
    
    bool TAG::getMetadata( const std::string& name, 
			   std::string& description, 
			   boost::posix_time::ptime& insertionTime, 
			   boost::posix_time::ptime& modificationTime, 
			   SessionImpl& session ){
      Query< DESCRIPTION, INSERTION_TIME, MODIFICATION_TIME > q( session.coralSchema() );
      q.addCondition<NAME>( name );
      for ( auto row : q ) std::tie( description, insertionTime, modificationTime ) = row;
      return q.retrievedRows();
    }
    
    void TAG::insert( const std::string& name, 
		      cond::TimeType timeType, 
		      const std::string& objectType, 
		      cond::SynchronizationType synchronizationType, 
		      cond::Time_t endOfValidity, 
		      const std::string& description, 
		      cond::Time_t lastValidatedTime, 
		      const boost::posix_time::ptime& insertionTime, 
		      SessionImpl& session  ){
      RowBuffer< NAME, TIME_TYPE, OBJECT_TYPE, SYNCHRONIZATION, END_OF_VALIDITY, DESCRIPTION, LAST_VALIDATED_TIME, INSERTION_TIME, MODIFICATION_TIME > 
	dataToInsert( std::tie( name, timeType, objectType, synchronizationType, endOfValidity, description, lastValidatedTime, insertionTime, insertionTime ) );
      insertInTable( session.coralSchema(), tname, dataToInsert.get() );
    }
    
    void TAG::update( const std::string& name, 
		      cond::Time_t& endOfValidity, 
		      const std::string& description, 
		      cond::Time_t lastValidatedTime,
		      const boost::posix_time::ptime& updateTime,  
		      SessionImpl& session ){
      UpdateBuffer buffer;
      buffer.setColumnData< END_OF_VALIDITY, DESCRIPTION, LAST_VALIDATED_TIME, MODIFICATION_TIME >( std::tie( endOfValidity, description, lastValidatedTime, updateTime  ) );
      buffer.addWhereCondition<NAME>( name );
      updateTable( session.coralSchema(), tname, buffer );  
    }
    
    void TAG::updateValidity( const std::string& name, 
			      cond::Time_t lastValidatedTime, 
			      const boost::posix_time::ptime& updateTime, 
			      SessionImpl& session ){
      UpdateBuffer buffer;
      buffer.setColumnData< LAST_VALIDATED_TIME, MODIFICATION_TIME >( std::tie( lastValidatedTime, updateTime  ) );
      buffer.addWhereCondition<NAME>( name );
      updateTable( session.coralSchema(), tname, buffer );
    }
    
    bool IOV::exists( SessionImpl& session ){
      return existsTable( session.coralSchema(), tname );
    }
    
    void IOV::create( SessionImpl& session ){
      if( exists( session )){
	throwException( "IOV table already exists in this schema.",
			"IOV::Schema::create");
      }
      
      TableDescription< TAG_NAME, SINCE, PAYLOAD_HASH, INSERTION_TIME > descr( tname );
      descr.setPrimaryKey< TAG_NAME, SINCE, INSERTION_TIME >();
      descr.setForeignKey< TAG_NAME, TAG::NAME >( "TAG_NAME_FK" );
      descr.setForeignKey< PAYLOAD_HASH, PAYLOAD::HASH >( "PAYLOAD_HASH_FK" );
      createTable( session.coralSchema(), descr.get() );
    }
    
    size_t IOV::selectGroups( const std::string& tag, std::vector<cond::Time_t>& groups, SessionImpl& session ){
      Query< SINCE_GROUP > q( session.coralSchema(), true );
      q.addCondition<TAG_NAME>( tag );
      q.addOrderClause<SINCE_GROUP>();
      for( auto row : q ){
	groups.push_back(std::get<0>(row));
      }
      return q.retrievedRows();
    }
    
    size_t IOV::selectSnapshotGroups( const std::string& tag, const boost::posix_time::ptime& snapshotTime, std::vector<cond::Time_t>& groups, SessionImpl& session ){
      Query< SINCE_GROUP > q( session.coralSchema(), true );
      q.addCondition<TAG_NAME>( tag );
      q.addCondition<INSERTION_TIME>( snapshotTime,"<=" );
      q.addOrderClause<SINCE_GROUP>();
      for( auto row : q ){
	groups.push_back(std::get<0>(row));
      }
      return q.retrievedRows();
    }
    
    size_t IOV::selectLastByGroup( const std::string& tag, cond::Time_t lowerSinceGroup, cond::Time_t upperSinceGroup , 
				   std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs, 
				   SessionImpl& session ){
      Query< SINCE, PAYLOAD_HASH > q( session.coralSchema() );
      q.addCondition<TAG_NAME>( tag );
      if( lowerSinceGroup > 0 ) q.addCondition<SINCE>( lowerSinceGroup, ">=" );
      if( upperSinceGroup < cond::time::MAX ) q.addCondition<SINCE>( upperSinceGroup, "<" );
      q.addOrderClause<SINCE>();
      q.addOrderClause<INSERTION_TIME>( false );
      size_t initialSize = iovs.size();
      for( auto row : q ){
	// starting from the second iov in the array, skip the rows with older timestamp
	if( iovs.size()-initialSize && std::get<0>(iovs.back()) == std::get<0>(row) ) continue;
	iovs.push_back( row );
      }
      return iovs.size()-initialSize;
    }
    
    size_t IOV::selectSnapshotByGroup( const std::string& tag, cond::Time_t lowerSinceGroup, cond::Time_t upperSinceGroup, 
				       const boost::posix_time::ptime& snapshotTime, 
				       std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs, 
				       SessionImpl& session ){
      Query< SINCE, PAYLOAD_HASH > q( session.coralSchema() );
      q.addCondition<TAG_NAME>( tag );
      if( lowerSinceGroup > 0 ) q.addCondition<SINCE>( lowerSinceGroup, ">=" );
      if( upperSinceGroup < cond::time::MAX ) q.addCondition<SINCE>( upperSinceGroup, "<" );
      q.addCondition<INSERTION_TIME>( snapshotTime,"<=" );
      q.addOrderClause<SINCE>();
      q.addOrderClause<INSERTION_TIME>( false );
      size_t initialSize = iovs.size();
      for ( auto row : q ) {
	// starting from the second iov in the array, skip the rows with older timestamp
	if( iovs.size()-initialSize && std::get<0>(iovs.back()) == std::get<0>(row) ) continue;
	iovs.push_back( row );
      }
      return iovs.size()-initialSize;
    }
    
    /**
size_t IOV::selectLastByGroup( const std::string& tag, 
cond::Time_t targetGroup, 
	       		       std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs, 
		       	       SessionImpl& session ){
  cond::Time_t lowerSinceGroup = 0; 
  // we should do better this (one query or caching this query result...)
  if( targetGroup ){
    Query< MAX_SINCE > q0( session.coralSchema() );
    q0.addCondition<TAG_NAME>( tag );
    q0.addCondition<SINCE>( targetGroup,"<=" );
    for( auto row : q0 ){
      lowerSinceGroup = std::get<0>(row);
    }
  }
  cond::Time_t upperSinceGroup = lowerSinceGroup+2*cond::time::SINCE_GROUP_SIZE;
  
  Query< SINCE, PAYLOAD_HASH > q1( session.coralSchema() );
  q1.addCondition<TAG_NAME>( tag );
  if( lowerSinceGroup > 0 ) q1.addCondition<SINCE>( lowerSinceGroup, ">=" );
  if( upperSinceGroup < cond::time::MAX ) q1.addCondition<SINCE>( upperSinceGroup, "<=" );
  q1.addOrderClause<SINCE>();
  q1.addOrderClause<INSERTION_TIME>( false );
  size_t initialSize = iovs.size();
  for( auto row : q1 ){
    // starting from the second iov in the array, skip the rows with older timestamp
    if( iovs.size()-initialSize && std::get<0>(iovs.back()) == std::get<0>(row) ) continue;
    iovs.push_back( row );
  }
  return iovs.size()-initialSize;
}

size_t IOV::selectSnapshotByGroup( const std::string& tag, 
	       			   cond::Time_t targetGroup, 
	       			   const boost::posix_time::ptime& snapshotUpperTime, 
		       		   std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs, 
	       			   SessionImpl& session ){
  cond::Time_t lowerSinceGroup = 0; 
  // we should do better this (one query or caching this query result...)
  if( targetGroup ){
    Query< MAX_SINCE > q0( session.coralSchema() );
    q0.addCondition<TAG_NAME>( tag );
    q0.addCondition<SINCE>( targetGroup,"<=" );
    for( auto row : q0 ){
      lowerSinceGroup = std::get<0>(row);
    }
  }
  cond::Time_t upperSinceGroup = lowerSinceGroup+2*cond::time::SINCE_GROUP_SIZE;

  Query< SINCE, PAYLOAD_HASH > q1( session.coralSchema() );
  q1.addCondition<TAG_NAME>( tag );
  if( lowerSinceGroup > 0 ) q1.addCondition<SINCE>( lowerSinceGroup, ">=" );
  if( upperSinceGroup < cond::time::MAX ) q1.addCondition<SINCE>( upperSinceGroup, "<=" );
  q1.addCondition<INSERTION_TIME>( snapshotUpperTime,"<=" );
  q1.addOrderClause<SINCE>();
  q1.addOrderClause<INSERTION_TIME>( false );
  size_t initialSize = iovs.size();
  for ( auto row : q1 ) {
    // starting from the second iov in the array, skip the rows with older timestamp
    if( iovs.size()-initialSize && std::get<0>(iovs.back()) == std::get<0>(row) ) continue;
    iovs.push_back( row );
  }
  return iovs.size()-initialSize;
}
**/

    size_t IOV::selectLast( const std::string& tag, 
			    std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs, 
			    SessionImpl& session ){
      Query< SINCE, PAYLOAD_HASH > q( session.coralSchema() );
      q.addCondition<TAG_NAME>( tag );
      q.addOrderClause<SINCE>();
      q.addOrderClause<INSERTION_TIME>( false );
      size_t initialSize = iovs.size();
      for ( auto row : q ) {
	// starting from the second iov in the array, skip the rows with older timestamp
	if( iovs.size()-initialSize && std::get<0>(iovs.back()) == std::get<0>(row) ) continue;
	iovs.push_back( row );
      }
      return iovs.size()-initialSize;
    }
    
    void IOV::insertOne( const std::string& tag, 
			 cond::Time_t since, 
			 cond::Hash payloadHash, 
			 const boost::posix_time::ptime& insertTimeStamp, 
			 SessionImpl& session){
      RowBuffer< TAG_NAME, SINCE, PAYLOAD_HASH, INSERTION_TIME > dataToInsert( std::tie( tag, since, payloadHash, insertTimeStamp ) );
      insertInTable( session.coralSchema(), tname, dataToInsert.get() );
    }
    
    void IOV::insertMany( const std::string& tag, 
			  const std::vector<std::tuple<cond::Time_t,cond::Hash,boost::posix_time::ptime> >& iovs, 
			  SessionImpl& session ){
      BulkInserter< TAG_NAME, SINCE, PAYLOAD_HASH, INSERTION_TIME > inserter( session.coralSchema(), tname );
      for( auto row : iovs ) inserter.insert( std::tuple_cat( std::tie(tag),row ) );
      
      inserter.flush();
    }
    
    bool PAYLOAD::exists( SessionImpl& session ){
      return existsTable( session.coralSchema(), tname );
    }
    
    void PAYLOAD::create( SessionImpl& session ){
      if( exists( session )){
	throwException( "Payload table already exists in this schema.",
			"PAYLOAD::Schema::create");
      }
      
      TableDescription< HASH, OBJECT_TYPE, DATA, STREAMER_INFO, VERSION, INSERTION_TIME > descr( tname );
      descr.setPrimaryKey<HASH>();
      createTable( session.coralSchema(), descr.get() );
    }
    
    bool PAYLOAD::select( const cond::Hash& payloadHash, 
			  SessionImpl& session ){
      Query< HASH > q( session.coralSchema() );
      q.addCondition<HASH>( payloadHash );
      for ( auto row : q ) {}
      
      return q.retrievedRows();

    }


    bool PAYLOAD::select( const cond::Hash& payloadHash, 
			  std::string& objectType, 
			  cond::Binary& payloadData, 
			  SessionImpl& session ){
      Query< DATA, OBJECT_TYPE > q( session.coralSchema() );
      q.addCondition<HASH>( payloadHash );
      for ( auto row : q ) {
	std::tie( payloadData, objectType ) = row;
      }
      return q.retrievedRows();
    }
    
    bool PAYLOAD::insert( const cond::Hash& payloadHash, 
			  const std::string& objectType,
			  const cond::Binary& payloadData, 				      
			  const boost::posix_time::ptime& insertionTime, 
			  SessionImpl& session ){
      cond::Binary dummy;
      std::string streamerType("ROOT5");
      dummy.copy( streamerType );
      std::string version("dummy");
      RowBuffer< HASH, OBJECT_TYPE, DATA, STREAMER_INFO, VERSION, INSERTION_TIME > dataToInsert( std::tie( payloadHash, objectType, payloadData, dummy, version, insertionTime ) ); 
      bool failOnDuplicate = false;
      return insertInTable( session.coralSchema(), tname, dataToInsert.get(), failOnDuplicate );
    }
    
    bool TAG_MIGRATION::exists( SessionImpl& session ){
      return existsTable( session.coralSchema(), tname );  
    }
    
    void TAG_MIGRATION::create( SessionImpl& session ){
      if( exists( session ) ){
	throwException( "TAG_MIGRATIONtable already exists in this schema.",
			"TAG::create");
      }
      TableDescription< SOURCE_ACCOUNT, SOURCE_TAG, TAG_NAME, INSERTION_TIME > descr( tname );
      descr.setPrimaryKey<SOURCE_ACCOUNT, SOURCE_TAG>();
      descr.setForeignKey< TAG_NAME, TAG::NAME >( "TAG_NAME_FK" );
      createTable( session.coralSchema(), descr.get() );
    }
    
    bool TAG_MIGRATION::select( const std::string& sourceAccount, const std::string& sourceTag, std::string& tagName, SessionImpl& session ){
      Query< TAG_NAME > q( session.coralSchema() );
      q.addCondition<SOURCE_ACCOUNT>( sourceAccount );
      q.addCondition<SOURCE_TAG>( sourceTag );
      for ( auto row : q ) {
	std::tie( tagName ) = row;
      }
      
      return q.retrievedRows();
      
    }
    
    void TAG_MIGRATION::insert( const std::string& sourceAccount, const std::string& sourceTag, const std::string& tagName, 
				const boost::posix_time::ptime& insertionTime, SessionImpl& session  ){
      RowBuffer< SOURCE_ACCOUNT, SOURCE_TAG, TAG_NAME, INSERTION_TIME > 
    dataToInsert( std::tie( sourceAccount, sourceTag, tagName, insertionTime ) );
      insertInTable( session.coralSchema(), tname, dataToInsert.get() );
    }
    
    bool iovDb::exists( SessionImpl& session ){
      size_t ntables = 0;
      if( TAG::exists( session ) ) ntables++;
      if( PAYLOAD::exists( session ) ) ntables++;
      if( IOV::exists( session ) ) ntables++;
      //if( ntables && ntables<3 ) throwException( "The IOV Database is incomplete or corrupted.","iovDb::exists" );
      //return ntables;
      return ntables==3;
    }
    
    bool iovDb::create( SessionImpl& session ){
      bool created = false;
      if( !exists( session ) ){
	TAG::create( session );
	PAYLOAD::create( session );
	IOV::create( session );
	created = true;
      }
      return created;
    }
    
  }
}

