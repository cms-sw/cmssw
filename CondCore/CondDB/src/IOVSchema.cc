#include "CondCore/CondDB/interface/Exception.h"
#include "IOVSchema.h"
//
#include <openssl/sha.h>

namespace cond {

  namespace persistency {

    cond::Hash makeHash( const std::string& objectType, const cond::Binary& data ){
      SHA_CTX ctx;
      if( !SHA1_Init( &ctx ) ){
	throwException( "SHA1 initialization error.","IOVSchema::makeHash");
      }
      if( !SHA1_Update( &ctx, objectType.c_str(), objectType.size() ) ){
	throwException( "SHA1 processing error (1).","IOVSchema::makeHash");
      }
      if( !SHA1_Update( &ctx, data.data(), data.size() ) ){
	throwException( "SHA1 processing error (2).","IOVSchema::makeHash");
      }
      unsigned char hash[SHA_DIGEST_LENGTH];
      if( !SHA1_Final(hash, &ctx) ){
	throwException( "SHA1 finalization error.","IOVSchema::makeHash");
      }
      
      char tmp[SHA_DIGEST_LENGTH*2+1];
      // re-write bytes in hex
      for (unsigned int i = 0; i < 20; i++) {                                                                                                        
	::sprintf(&tmp[i * 2], "%02x", hash[i]);                                                                                                 
      }                                                                                                                                              
      tmp[20*2] = 0;                                                                                                                                 
      return tmp;                                                                                                                                    
    }

    TAG::Table::Table( coral::ISchema& schema ):
      m_schema( schema ){
    }

    bool TAG::Table::Table::exists(){
      return existsTable( m_schema, tname );
    }
   
    void TAG::Table::create(){
      if( exists() ){
	throwException( "TAG table already exists in this schema.",
			"TAG::Table::create");
      }
      TableDescription< NAME, TIME_TYPE, OBJECT_TYPE, SYNCHRONIZATION, END_OF_VALIDITY, DESCRIPTION, LAST_VALIDATED_TIME, INSERTION_TIME, MODIFICATION_TIME > descr( tname );
      descr.setPrimaryKey<NAME>();
      createTable( m_schema, descr.get() );
    }
    
    bool TAG::Table::select( const std::string& name ){
      Query< NAME > q( m_schema );
      q.addCondition<NAME>( name );
      for ( auto row : q ) {}
      
      return q.retrievedRows();
    }
    
    bool TAG::Table::select( const std::string& name, 
			     cond::TimeType& timeType, 
			     std::string& objectType, 
			     cond::SynchronizationType& synchronizationType,
			     cond::Time_t& endOfValidity,
			     std::string& description, 
			     cond::Time_t&  lastValidatedTime ){
      Query< TIME_TYPE, OBJECT_TYPE, SYNCHRONIZATION, END_OF_VALIDITY, DESCRIPTION, LAST_VALIDATED_TIME > q( m_schema );
      q.addCondition<NAME>( name );
      for ( auto row : q ) std::tie( timeType, objectType, synchronizationType, endOfValidity, description, lastValidatedTime ) = row;
      
      return q.retrievedRows();
    }
    
    bool TAG::Table::getMetadata( const std::string& name, 
				  std::string& description, 
				  boost::posix_time::ptime& insertionTime, 
				  boost::posix_time::ptime& modificationTime ){
      Query< DESCRIPTION, INSERTION_TIME, MODIFICATION_TIME > q( m_schema );
      q.addCondition<NAME>( name );
      for ( auto row : q ) std::tie( description, insertionTime, modificationTime ) = row;
      return q.retrievedRows();
    }
    
    void TAG::Table::insert( const std::string& name, 
			     cond::TimeType timeType, 
			     const std::string& objectType, 
			     cond::SynchronizationType synchronizationType, 
			     cond::Time_t endOfValidity, 
			     const std::string& description, 
			     cond::Time_t lastValidatedTime, 
			     const boost::posix_time::ptime& insertionTime ){
      RowBuffer< NAME, TIME_TYPE, OBJECT_TYPE, SYNCHRONIZATION, END_OF_VALIDITY, DESCRIPTION, LAST_VALIDATED_TIME, INSERTION_TIME, MODIFICATION_TIME > 
	dataToInsert( std::tie( name, timeType, objectType, synchronizationType, endOfValidity, description, lastValidatedTime, insertionTime, insertionTime ) );
      insertInTable( m_schema, tname, dataToInsert.get() );
    }
    
    void TAG::Table::update( const std::string& name, 
		      cond::Time_t& endOfValidity, 
		      const std::string& description, 
		      cond::Time_t lastValidatedTime,
		      const boost::posix_time::ptime& updateTime ){
      UpdateBuffer buffer;
      buffer.setColumnData< END_OF_VALIDITY, DESCRIPTION, LAST_VALIDATED_TIME, MODIFICATION_TIME >( std::tie( endOfValidity, description, lastValidatedTime, updateTime  ) );
      buffer.addWhereCondition<NAME>( name );
      updateTable( m_schema, tname, buffer );  
    }
    
    void TAG::Table::updateValidity( const std::string& name, 
				     cond::Time_t lastValidatedTime, 
				     const boost::posix_time::ptime& updateTime ){
      UpdateBuffer buffer;
      buffer.setColumnData< LAST_VALIDATED_TIME, MODIFICATION_TIME >( std::tie( lastValidatedTime, updateTime  ) );
      buffer.addWhereCondition<NAME>( name );
      updateTable( m_schema, tname, buffer );
    }
    
    IOV::Table::Table( coral::ISchema& schema ):
      m_schema( schema ){
    }

    bool IOV::Table::exists(){
      return existsTable( m_schema, tname );
    }
    
    void IOV::Table::create(){
      if( exists()){
	throwException( "IOV table already exists in this schema.",
			"IOV::Schema::create");
      }
      
      TableDescription< TAG_NAME, SINCE, PAYLOAD_HASH, INSERTION_TIME > descr( tname );
      descr.setPrimaryKey< TAG_NAME, SINCE, INSERTION_TIME >();
      descr.setForeignKey< TAG_NAME, TAG::NAME >( "TAG_NAME_FK" );
      descr.setForeignKey< PAYLOAD_HASH, PAYLOAD::HASH >( "PAYLOAD_HASH_FK" );
      createTable( m_schema, descr.get() );
    }
    
    size_t IOV::Table::selectGroups( const std::string& tag, std::vector<cond::Time_t>& groups ){
      Query< SINCE_GROUP > q( m_schema, true );
      q.addCondition<TAG_NAME>( tag );
      q.addOrderClause<SINCE_GROUP>();
      for( auto row : q ){
	groups.push_back(std::get<0>(row));
      }
      return q.retrievedRows();
    }
    
    size_t IOV::Table::selectSnapshotGroups( const std::string& tag, const boost::posix_time::ptime& snapshotTime, std::vector<cond::Time_t>& groups ){
      Query< SINCE_GROUP > q( m_schema, true );
      q.addCondition<TAG_NAME>( tag );
      q.addCondition<INSERTION_TIME>( snapshotTime,"<=" );
      q.addOrderClause<SINCE_GROUP>();
      for( auto row : q ){
	groups.push_back(std::get<0>(row));
      }
      return q.retrievedRows();
    }
    
    size_t IOV::Table::selectLatestByGroup( const std::string& tag, cond::Time_t lowerSinceGroup, cond::Time_t upperSinceGroup , 
					    std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      Query< SINCE, PAYLOAD_HASH > q( m_schema );
      q.addCondition<TAG_NAME>( tag );
      if( lowerSinceGroup > 0 ) q.addCondition<SINCE>( lowerSinceGroup, ">=" );
      if( upperSinceGroup < cond::time::MAX_VAL ) q.addCondition<SINCE>( upperSinceGroup, "<" );
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
    
    size_t IOV::Table::selectSnapshotByGroup( const std::string& tag, cond::Time_t lowerSinceGroup, cond::Time_t upperSinceGroup, 
					      const boost::posix_time::ptime& snapshotTime, 
					      std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      Query< SINCE, PAYLOAD_HASH > q( m_schema );
      q.addCondition<TAG_NAME>( tag );
      if( lowerSinceGroup > 0 ) q.addCondition<SINCE>( lowerSinceGroup, ">=" );
      if( upperSinceGroup < cond::time::MAX_VAL ) q.addCondition<SINCE>( upperSinceGroup, "<" );
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
    
    size_t IOV::Table::selectLatest( const std::string& tag, 
				     std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      Query< SINCE, PAYLOAD_HASH > q( m_schema );
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

    bool IOV::Table::getLastIov( const std::string& tag, cond::Time_t& since, cond::Hash& hash ){
      Query< SINCE, PAYLOAD_HASH > q( m_schema );
      q.addCondition<TAG_NAME>( tag );
      q.addOrderClause<SINCE>( false );
      q.addOrderClause<INSERTION_TIME>( false );
      for ( auto row : q ) {
	since = std::get<0>(row);
	hash = std::get<1>(row);
	return true;
      }
      return false;
    }

    bool IOV::Table::getSize( const std::string& tag, size_t& size ){
      Query< SEQUENCE_SIZE > q( m_schema );
      q.addCondition<TAG_NAME>( tag );
      for ( auto row : q ) {
	size = std::get<0>(row);
	return true;
      }
      return false;
    }
    
    bool IOV::Table::getSnapshotSize( const std::string& tag, const boost::posix_time::ptime& snapshotTime, size_t& size ){
      Query< SEQUENCE_SIZE > q( m_schema );
      q.addCondition<TAG_NAME>( tag );
      q.addCondition<INSERTION_TIME>( snapshotTime,"<=" );
      for ( auto row : q ) {
	size = std::get<0>(row);
	return true;
      }
      return false;
    }

    void IOV::Table::insertOne( const std::string& tag, 
				cond::Time_t since, 
				cond::Hash payloadHash, 
				const boost::posix_time::ptime& insertTimeStamp ){
      RowBuffer< TAG_NAME, SINCE, PAYLOAD_HASH, INSERTION_TIME > dataToInsert( std::tie( tag, since, payloadHash, insertTimeStamp ) );
      insertInTable( m_schema, tname, dataToInsert.get() );
    }
    
    void IOV::Table::insertMany( const std::string& tag, 
				 const std::vector<std::tuple<cond::Time_t,cond::Hash,boost::posix_time::ptime> >& iovs ){
      BulkInserter< TAG_NAME, SINCE, PAYLOAD_HASH, INSERTION_TIME > inserter( m_schema, tname );
      for( auto row : iovs ) inserter.insert( std::tuple_cat( std::tie(tag),row ) );
      
      inserter.flush();
    }
    
    PAYLOAD::Table::Table( coral::ISchema& schema ):
      m_schema( schema ){
    }

    bool PAYLOAD::Table::exists(){
      return existsTable( m_schema, tname );
    }
    
    void PAYLOAD::Table::create(){
      if( exists()){
	throwException( "Payload table already exists in this schema.",
			"PAYLOAD::Schema::create");
      }
      
      TableDescription< HASH, OBJECT_TYPE, DATA, STREAMER_INFO, VERSION, INSERTION_TIME > descr( tname );
      descr.setPrimaryKey<HASH>();
      createTable( m_schema, descr.get() );
    }
    
    bool PAYLOAD::Table::select( const cond::Hash& payloadHash ){
      Query< HASH > q( m_schema );
      q.addCondition<HASH>( payloadHash );
      for ( auto row : q ) {}
      
      return q.retrievedRows();
    }

    bool PAYLOAD::Table::getType( const cond::Hash& payloadHash, std::string& objectType ){
      Query< OBJECT_TYPE > q( m_schema );
      q.addCondition<HASH>( payloadHash );
      for ( auto row : q ) {
	objectType = std::get<0>(row);
      }
      
      return q.retrievedRows(); 
    }

    bool PAYLOAD::Table::select( const cond::Hash& payloadHash, 
				 std::string& objectType, 
				 cond::Binary& payloadData,
				 cond::Binary& streamerInfoData ){
      Query< DATA, STREAMER_INFO, OBJECT_TYPE > q( m_schema );
      q.addCondition<HASH>( payloadHash );
      for ( auto row : q ) {
	std::tie( payloadData, streamerInfoData, objectType ) = row;
      }
      return q.retrievedRows();
    }
    
    bool PAYLOAD::Table::insert( const cond::Hash& payloadHash, 
    				 const std::string& objectType,
    				 const cond::Binary& payloadData, 
				 const cond::Binary& streamerInfoData,				      
    				 const boost::posix_time::ptime& insertionTime ){
      std::string version("dummy");
      RowBuffer< HASH, OBJECT_TYPE, DATA, STREAMER_INFO, VERSION, INSERTION_TIME > dataToInsert( std::tie( payloadHash, objectType, payloadData, streamerInfoData, version, insertionTime ) ); 
      bool failOnDuplicate = false;
      return insertInTable( m_schema, tname, dataToInsert.get(), failOnDuplicate );
    }

    cond::Hash PAYLOAD::Table::insertIfNew( const std::string& payloadObjectType, 
					    const cond::Binary& payloadData, 
					    const cond::Binary& streamerInfoData,
					    const boost::posix_time::ptime& insertionTime ){
      cond::Hash payloadHash = makeHash( payloadObjectType, payloadData );
      // the check on the hash existance is only required to avoid the error message printing in SQLite! once this is removed, this check is useless... 
      if( !select( payloadHash ) ){
	insert( payloadHash, payloadObjectType, payloadData, streamerInfoData, insertionTime );
      }
      return payloadHash;
    }
    
    TAG_MIGRATION::Table::Table( coral::ISchema& schema ):
      m_schema( schema ){
    }

    bool TAG_MIGRATION::Table::exists(){
      return existsTable( m_schema, tname );  
    }
    
    void TAG_MIGRATION::Table::create(){
      if( exists() ){
	throwException( "TAG_MIGRATIONtable already exists in this schema.",
			"TAG::create");
      }
      TableDescription< SOURCE_ACCOUNT, SOURCE_TAG, TAG_NAME, INSERTION_TIME > descr( tname );
      descr.setPrimaryKey<SOURCE_ACCOUNT, SOURCE_TAG>();
      descr.setForeignKey< TAG_NAME, TAG::NAME >( "TAG_NAME_FK" );
      createTable( m_schema, descr.get() );
    }
    
    bool TAG_MIGRATION::Table::select( const std::string& sourceAccount, const std::string& sourceTag, std::string& tagName ){
      Query< TAG_NAME > q( m_schema );
      q.addCondition<SOURCE_ACCOUNT>( sourceAccount );
      q.addCondition<SOURCE_TAG>( sourceTag );
      for ( auto row : q ) {
	std::tie( tagName ) = row;
      }
      
      return q.retrievedRows();
      
    }
    
    void TAG_MIGRATION::Table::insert( const std::string& sourceAccount, const std::string& sourceTag, const std::string& tagName, 
				const boost::posix_time::ptime& insertionTime ){
      RowBuffer< SOURCE_ACCOUNT, SOURCE_TAG, TAG_NAME, INSERTION_TIME > 
    dataToInsert( std::tie( sourceAccount, sourceTag, tagName, insertionTime ) );
      insertInTable( m_schema, tname, dataToInsert.get() );
    }
    
    IOVSchema::IOVSchema( coral::ISchema& schema ):
      m_tagTable( schema ),
      m_iovTable( schema ),
      m_payloadTable( schema ),
      m_tagMigrationTable( schema ){
    }
      
    bool IOVSchema::exists(){
      if( !m_tagTable.exists() ) return false;
      if( !m_payloadTable.exists() ) return false;
      if( !m_iovTable.exists() ) return false;
      return true;
    }
    
    bool IOVSchema::create(){
      bool created = false;
      if( !exists() ){
	m_tagTable.create();
	m_payloadTable.create();
	m_iovTable.create();
	created = true;
      }
      return created;
    }

    ITagTable& IOVSchema::tagTable(){
      return m_tagTable;
    }
      
    IIOVTable& IOVSchema::iovTable(){
      return m_iovTable;
    }
      
    IPayloadTable& IOVSchema::payloadTable(){
      return m_payloadTable;
    }
      
    ITagMigrationTable& IOVSchema::tagMigrationTable(){
      return m_tagMigrationTable;
    }
    
  }
}

