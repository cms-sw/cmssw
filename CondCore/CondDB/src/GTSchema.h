#ifndef CondCore_CondDB_GTSchema_h
#define CondCore_CondDB_GTSchema_h

#include "DbCore.h"
#include "IDbSchema.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {
    
    conddb_table( GLOBAL_TAG ) {
      
      conddb_column( NAME, std::string );
      conddb_column( VALIDITY, cond::Time_t );
      conddb_column( DESCRIPTION, std::string );
      conddb_column( RELEASE, std::string );
      conddb_column( SNAPSHOT_TIME, boost::posix_time::ptime );
      conddb_column( INSERTION_TIME, boost::posix_time::ptime );
      
      class Table : public IGTTable {
      public:
	explicit Table( coral::ISchema& schema );
	virtual ~Table(){}
	bool exists();
	void create();
	bool select( const std::string& name);
	bool select( const std::string& name, cond::Time_t& validity, boost::posix_time::ptime& snapshotTime );
	bool select( const std::string& name, cond::Time_t& validity, std::string& description, 
		     std::string& release, boost::posix_time::ptime& snapshotTime );
	void insert( const std::string& name, cond::Time_t validity, const std::string& description, const std::string& release, 
		     const boost::posix_time::ptime& snapshotTime, const boost::posix_time::ptime& insertionTime );
	void update( const std::string& name, cond::Time_t validity, const std::string& description, const std::string& release, 
		     const boost::posix_time::ptime& snapshotTime, const boost::posix_time::ptime& insertionTime );
      private:
	coral::ISchema& m_schema;
      };
    }
    
    conddb_table ( GLOBAL_TAG_MAP ) {
      
      static constexpr unsigned int PAYLOAD_HASH_SIZE = 40;
      
      conddb_column( GLOBAL_TAG_NAME, std::string );
      // to be changed to RECORD_NAME!
      conddb_column( RECORD, std::string );
      // to be changed to RECORD_LABEL!
      conddb_column( LABEL, std::string );
      conddb_column( TAG_NAME, std::string );
      
      class Table : public IGTMapTable {
      public:
	explicit Table( coral::ISchema& schema );
	virtual ~Table(){}
	bool exists();
	void create();
	bool select( const std::string& gtName, std::vector<std::tuple<std::string,std::string,std::string> >& tags );
	bool select( const std::string& gtName, const std::string& preFix, const std::string& postFix,
		     std::vector<std::tuple<std::string,std::string,std::string> >& tags );
	void insert( const std::string& gtName, const std::vector<std::tuple<std::string,std::string,std::string> >& tags );
      private:
	coral::ISchema& m_schema;
      };
    }
    
    class GTSchema : public IGTSchema {
    public: 
      explicit GTSchema( coral::ISchema& schema );
      virtual ~GTSchema(){}
      bool exists();
      void create();
      GLOBAL_TAG::Table& gtTable();
      GLOBAL_TAG_MAP::Table& gtMapTable();
    private:
      GLOBAL_TAG::Table m_gtTable;
      GLOBAL_TAG_MAP::Table m_gtMapTable;
    };
    
  }
}
#endif
