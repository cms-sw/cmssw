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
			     cond::SynchronizationType synchronizationType,
			     cond::Time_t& endOfValidity, 
			     const std::string& description, 
			     cond::Time_t lastValidatedTime,
			     const boost::posix_time::ptime& updateTime ){
      UpdateBuffer buffer;
      buffer.setColumnData< SYNCHRONIZATION, END_OF_VALIDITY, DESCRIPTION, LAST_VALIDATED_TIME, MODIFICATION_TIME >( std::tie( synchronizationType, 
															       endOfValidity, 
															       description, 
															       lastValidatedTime, 
															       updateTime  ) );
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
      q.groupBy(SINCE_GROUP::group());
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
      q.groupBy(SINCE_GROUP::group());
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

    size_t IOV::Table::selectSnapshot( const std::string& tag,
                                       const boost::posix_time::ptime& snapshotTime,
                                       std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs){
      Query< SINCE, PAYLOAD_HASH > q( m_schema );
      q.addCondition<TAG_NAME>( tag );
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

    bool IOV::Table::getSnapshotLastIov( const std::string& tag, const boost::posix_time::ptime& snapshotTime, cond::Time_t& since, cond::Hash& hash ){
      Query< SINCE, PAYLOAD_HASH > q( m_schema );
      q.addCondition<TAG_NAME>( tag );
      q.addCondition<INSERTION_TIME>( snapshotTime,"<=" );
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

    bool IOV::Table::getRange( const std::string& tag, cond::Time_t begin, cond::Time_t end, 
			       std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      Query< MAX_SINCE > q0( m_schema );
      q0.addCondition<TAG_NAME>( tag );
      q0.addCondition<SINCE>( begin, "<=");
      cond::Time_t lower_since = 0;
      for( auto row: q0 ){
	lower_since = std::get<0>( row );
      }
      if ( !lower_since ) return false;
      Query< SINCE, PAYLOAD_HASH > q1( m_schema );
      q1.addCondition<TAG_NAME>( tag );
      q1.addCondition<SINCE>( lower_since, ">=");
      q1.addCondition<SINCE>( end, "<=");
      for( auto row: q1 ){
        iovs.push_back( row );
      }
      return iovs.size();
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

    void IOV::Table::erase( const std::string& tag ){
      DeleteBuffer buffer;
      buffer.addWhereCondition<TAG_NAME>( tag );
      deleteFromTable( m_schema, tname, buffer );  
    }
    
    TAG_LOG::Table::Table( coral::ISchema& schema ):
      m_schema( schema ){
    }

    bool TAG_LOG::Table::exists(){
      return existsTable( m_schema, tname );  
    }

    void TAG_LOG::Table::create(){
      if( exists() ){
	throwException( "TAG_LOG table already exists in this schema.",
			"TAG_LOG::create");
      }
      TableDescription< TAG_NAME, EVENT_TIME, USER_NAME, HOST_NAME, COMMAND, ACTION, USER_TEXT > descr( tname );
      descr.setPrimaryKey<TAG_NAME, EVENT_TIME, ACTION>();
      descr.setForeignKey< TAG_NAME, TAG::NAME >( "TAG_NAME_FK" );
      createTable( m_schema, descr.get() );
    }

    void TAG_LOG::Table::insert( const std::string& tag, 
				 const boost::posix_time::ptime& eventTime, 
				 const std::string& userName,
				 const std::string& hostName,
				 const std::string& command,
				 const std::string& action,
				 const std::string& userText){
      RowBuffer< TAG_NAME, EVENT_TIME, USER_NAME, HOST_NAME, COMMAND, ACTION, USER_TEXT > 
	dataToInsert( std::tie( tag, eventTime, userName, hostName, command, action, userText ) );
      insertInTable( m_schema, tname, dataToInsert.get() );      
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
      cond::Binary sinfoData( streamerInfoData );
      if( !sinfoData.size() ) sinfoData.copy( std::string("0") );   
      RowBuffer< HASH, OBJECT_TYPE, DATA, STREAMER_INFO, VERSION, INSERTION_TIME > dataToInsert( std::tie( payloadHash, objectType, payloadData, sinfoData, version, insertionTime ) ); 
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
    
    IOVSchema::IOVSchema( coral::ISchema& schema ):
      m_tagTable( schema ),
      m_iovTable( schema ),
      m_tagLogTable( schema ),
      m_payloadTable( schema ){
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
	m_tagLogTable.create();
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
      
    ITagLogTable& IOVSchema::tagLogTable(){
      return m_tagLogTable;
    }

    IPayloadTable& IOVSchema::payloadTable(){
      return m_payloadTable;
    }

   
    bool runinfo::getRunStartTime( coral::ISchema& schema, 
				   cond::Time_t start, 
				   cond::Time_t end, 
				   std::vector<std::tuple<cond::Time_t,boost::posix_time::ptime> >& runData ){
      static constexpr const char* const RUNINFO_DATE_T = "RUNSESSION_DATE";
      static constexpr const char* const RUNINFO_PARAMETER_T = "RUNSESSION_PARAMETER";
      static constexpr const char* const RUNINFO_DATE_VALUE_C = "RUNSESSION_DATE.VALUE";
      static constexpr const char* const RUNINFO_PARAMETER_RUNNUM_C = "RUNSESSION_PARAMETER.RUNNUMBER";
      static constexpr const char* const RUNNUM_C = "RUNNUMBER";
      static constexpr const char* const ID_C = "ID";
      static constexpr const char* const MIN_RUNINFO_PARAMETER_RUNNUM_C = "MIN(RUNSESSION_PARAMETER.RUNNUMBER)";
      // first find the lowest run in the range
      bool ret = false;
      std::unique_ptr<coral::IQuery> q0( schema.newQuery() );
      q0->addToTableList( RUNINFO_PARAMETER_T );
      q0->addToOutputList( MIN_RUNINFO_PARAMETER_RUNNUM_C );
      q0->defineOutputType( MIN_RUNINFO_PARAMETER_RUNNUM_C, coral::AttributeSpecification::typeNameForType<int>() );
      static constexpr const char* const RUNNUMBER_VAL = "RUNNUMBER_VAL";
      std::string whereCl0( RUNINFO_PARAMETER_RUNNUM_C + std::string(" >= :") + RUNNUMBER_VAL );
      coral::AttributeList whereData0;
      whereData0.extend<int>( RUNNUMBER_VAL );
      whereData0[  RUNNUMBER_VAL ].data<int>() = end;
      q0->setCondition( whereCl0, whereData0 );
      coral::ICursor& res0 = q0->execute();
      if( res0.next() ){
        ret = true;
	const coral::AttributeList& row = res0.currentRow();
        end = row[ MIN_RUNINFO_PARAMETER_RUNNUM_C ].data<int>();
      }
      // then join the DATE and PARAM tables and retrieve all the runs in the range, with their start times
      // requires optimization...
      std::unique_ptr<coral::IQuery> q1( schema.newQuery() );
      static constexpr const char* const SELECTED_RUNS = "SELECTED_RUNS";
      auto& sq = q1->defineSubQuery( SELECTED_RUNS );
      sq.addToTableList( RUNINFO_PARAMETER_T );
      sq.addToOutputList( RUNNUM_C );
      sq.addToOutputList( ID_C );
      std::string startRunVar("START_RUN");
      std::string endRunVar("END_RUN");
      std::string whereCl1(  RUNNUM_C + std::string(">=:")+startRunVar+std::string(" AND ") +
			     RUNNUM_C + std::string("<=:")+endRunVar +std::string(" AND ")+
			     std::string("NAME='CMS.LVL0:START_TIME_T'") );
      coral::AttributeList whereData1;
      whereData1.extend<int>( startRunVar );
      whereData1.extend<int>( endRunVar );
      whereData1[ startRunVar ].data<int>() = start;
      whereData1[ endRunVar ].data<int>() = end;
      sq.setCondition( whereCl1, whereData1 );
      q1->addToTableList( SELECTED_RUNS );
      q1->addToTableList( RUNINFO_DATE_T );
      q1->addToOutputList( RUNINFO_DATE_VALUE_C );
      q1->addToOutputList( std::string(SELECTED_RUNS)+"."+RUNNUM_C );
      q1->defineOutputType( RUNINFO_DATE_VALUE_C, coral::AttributeSpecification::typeNameForType<coral::TimeStamp>() );
      q1->defineOutputType( std::string(SELECTED_RUNS)+"."+RUNNUM_C, coral::AttributeSpecification::typeNameForType<int>() );
      std::string whereCl2( std::string(SELECTED_RUNS)+"."+ID_C+" = RUNSESSION_DATE.RUNSESSION_PARAMETER_ID");
      q1->setCondition( whereCl2, coral::AttributeList() );
      /**
      q1->addToTableList( RUNINFO_DATE_T );
      q1->addToTableList( RUNINFO_PARAMETER_T );
      q1->addToOutputList( RUNINFO_DATE_VALUE_C );
      q1->addToOutputList( RUNINFO_PARAMETER_RUNNUM_C );
      q1->defineOutputType( RUNINFO_DATE_VALUE_C, coral::AttributeSpecification::typeNameForType<coral::TimeStamp>() );
      q1->defineOutputType( RUNINFO_PARAMETER_RUNNUM_C, coral::AttributeSpecification::typeNameForType<int>() );
      std::string startRunVar("START_RUN");
      std::string endRunVar("END_RUN");
      std::string whereCl1(  RUNINFO_PARAMETER_RUNNUM_C + std::string(">=:")+startRunVar+std::string(" AND ") +
			     RUNINFO_PARAMETER_RUNNUM_C + std::string("<=:")+endRunVar +std::string(" AND ")+
			     std::string("RUNSESSION_PARAMETER.NAME='CMS.LVL0:START_TIME_T' AND ")+
			     std::string("RUNSESSION_PARAMETER.ID=RUNSESSION_DATE.RUNSESSION_PARAMETER_ID") );
      coral::AttributeList whereData1;
      whereData1.extend<int>( startRunVar );
      whereData1.extend<int>( endRunVar );
      whereData1[ startRunVar ].data<int>() = start;
      whereData1[ endRunVar ].data<int>() = end;
      q1->setCondition( whereCl1, whereData1 );
      **/
      coral::ICursor& res1 = q1->execute();
      while( res1.next() ){
	const coral::AttributeList& row = res1.currentRow();
	//runData.push_back( std::make_tuple( row[ RUNINFO_PARAMETER_RUNNUM_C ].data<int>() , row[ RUNINFO_DATE_VALUE_C ].data<coral::TimeStamp>().time() ) );
	runData.push_back( std::make_tuple( row[ std::string(SELECTED_RUNS)+"."+RUNNUM_C ].data<int>() , row[ RUNINFO_DATE_VALUE_C ].data<coral::TimeStamp>().time() ) );
      }
      return ret;
    }
      
  }
}

