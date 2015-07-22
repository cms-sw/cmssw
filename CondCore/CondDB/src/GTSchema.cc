#include "CondCore/CondDB/interface/Exception.h"
#include "GTSchema.h"
//
namespace cond {

  namespace persistency {

    GLOBAL_TAG::Table::Table( coral::ISchema& schema ):
      m_schema( schema ){
    }

    bool GLOBAL_TAG::Table::exists(){
      return existsTable( m_schema, tname );
    }

    void GLOBAL_TAG::Table::create(){
      if( exists() ){
	throwException( "GLOBAL_TAG table already exists in this schema.",
			"GLOBAL_TAG::Table::create");
      }
      TableDescription< NAME, VALIDITY, DESCRIPTION, RELEASE, SNAPSHOT_TIME, INSERTION_TIME > descr( tname );
      descr.setPrimaryKey<NAME>();
      createTable( m_schema, descr.get() );
    }

    bool GLOBAL_TAG::Table::select( const std::string& name ){
      Query< NAME > q( m_schema );
      q.addCondition<NAME>( name );
      for ( auto row : q ) {}
      
      return q.retrievedRows();
    }
    
    bool GLOBAL_TAG::Table::select( const std::string& name, 
				    cond::Time_t& validity, 
				    boost::posix_time::ptime& snapshotTime ){
      Query< VALIDITY, SNAPSHOT_TIME > q( m_schema );
      q.addCondition<NAME>( name );
      for ( auto row : q ) std::tie( validity, snapshotTime ) = row;
      
      return q.retrievedRows();
    }
    
    bool GLOBAL_TAG::Table::select( const std::string& name, 
				    cond::Time_t& validity, 
				    std::string& description, 
				    std::string& release, 
				    boost::posix_time::ptime& snapshotTime ){
      Query< VALIDITY, DESCRIPTION, RELEASE, SNAPSHOT_TIME > q( m_schema );
      q.addCondition<NAME>( name );
      for ( auto row : q ) std::tie( validity, description, release, snapshotTime ) = row;

      return q.retrievedRows();
    }
    
    void GLOBAL_TAG::Table::insert( const std::string& name, 
				    cond::Time_t validity, 
				    const std::string& description, 
				    const std::string& release, 
				    const boost::posix_time::ptime& snapshotTime, 
				    const boost::posix_time::ptime& insertionTime ){
      RowBuffer< NAME, VALIDITY, DESCRIPTION, RELEASE, SNAPSHOT_TIME, INSERTION_TIME > 
	dataToInsert( std::tie( name, validity, description, release, snapshotTime, insertionTime ) );
      insertInTable( m_schema, tname, dataToInsert.get() );
    }
    
    void GLOBAL_TAG::Table::update( const std::string& name, 
				    cond::Time_t validity, 
				    const std::string& description, 
				    const std::string& release, 
				    const boost::posix_time::ptime& snapshotTime, 
				    const boost::posix_time::ptime& insertionTime ){
      UpdateBuffer buffer;
      buffer.setColumnData< VALIDITY, DESCRIPTION, RELEASE, SNAPSHOT_TIME, INSERTION_TIME >( std::tie( validity, description, release, snapshotTime, insertionTime  ) );
      buffer.addWhereCondition<NAME>( name );
      updateTable( m_schema, tname, buffer );  
    }
    
    GLOBAL_TAG_MAP::Table::Table( coral::ISchema& schema ):
      m_schema( schema ){
    }

    bool GLOBAL_TAG_MAP::Table::exists(){
      return existsTable( m_schema, tname );
    }

    void GLOBAL_TAG_MAP::Table::create(){
      if( exists() ){
	throwException( "GLOBAL_TAG_MAP table already exists in this schema.",
			"GLOBAL_TAG_MAP::Table::create");
      }
      TableDescription< GLOBAL_TAG_NAME, RECORD, LABEL, TAG_NAME > descr( tname );
      descr.setPrimaryKey< GLOBAL_TAG_NAME, RECORD, LABEL >();
      createTable( m_schema, descr.get() );
    }
    
    bool GLOBAL_TAG_MAP::Table::select( const std::string& gtName, 
					std::vector<std::tuple<std::string,std::string,std::string> >& tags ){
      Query< RECORD, LABEL, TAG_NAME > q( m_schema );
      q.addCondition< GLOBAL_TAG_NAME >( gtName );
      q.addOrderClause<RECORD>();
      q.addOrderClause<LABEL>();
      for ( auto row : q ) {
	if ( std::get<1>(row) == "-" ) {
	  std::get<1>(row) = "";
	}
	tags.push_back( row );
      }
      return q.retrievedRows();
    }
    
    bool GLOBAL_TAG_MAP::Table::select( const std::string& gtName, const std::string&, const std::string&,
					std::vector<std::tuple<std::string,std::string,std::string> >& tags ){
      return select( gtName, tags );
    }

    void GLOBAL_TAG_MAP::Table::insert( const std::string& gtName, 
					const std::vector<std::tuple<std::string,std::string,std::string> >& tags ){
      BulkInserter<GLOBAL_TAG_NAME, RECORD, LABEL, TAG_NAME > inserter( m_schema, tname );
      for( auto row : tags ) inserter.insert( std::tuple_cat( std::tie( gtName ),row ) );
      inserter.flush();  
    }
    
    GTSchema::GTSchema( coral::ISchema& schema ):
      m_gtTable( schema ),
      m_gtMapTable( schema ){
    }

    bool GTSchema::exists(){
      if( !m_gtTable.exists() ) return false;
      if( !m_gtMapTable.exists() ) return false;
      return true;
    }

    void GTSchema::create(){
      m_gtTable.create();
      m_gtMapTable.create();
    }

    GLOBAL_TAG::Table& GTSchema::gtTable(){
      return m_gtTable;
    }
      
    GLOBAL_TAG_MAP::Table& GTSchema::gtMapTable(){
      return m_gtMapTable;
    }
    
  }
}
