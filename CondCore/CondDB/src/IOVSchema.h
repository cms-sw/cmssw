#ifndef CondCore_CondDB_IOVSchema_h
#define CondCore_CondDB_IOVSchema_h

#include "DbCore.h"
#include "IDbSchema.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {

    conddb_table( TAG ) {
      
      conddb_column( NAME, std::string );
      conddb_column( TIME_TYPE, cond::TimeType );
      conddb_column( OBJECT_TYPE, std::string );
      conddb_column( SYNCHRONIZATION, cond::SynchronizationType );
      conddb_column( END_OF_VALIDITY, cond::Time_t );
      conddb_column( DESCRIPTION, std::string );
      conddb_column( LAST_VALIDATED_TIME, cond::Time_t );
      conddb_column( INSERTION_TIME, boost::posix_time::ptime );
      conddb_column( MODIFICATION_TIME, boost::posix_time::ptime );
      
      class Table : public ITagTable {
      public:
	explicit Table( coral::ISchema& schema );
	virtual ~Table(){}
	bool exists();
	void create();
	bool select( const std::string& name );
	bool select( const std::string& name, cond::TimeType& timeType, std::string& objectType, cond::SynchronizationType& synchronizationType,
		     cond::Time_t& endOfValidity, std::string& description, cond::Time_t& lastValidatedTime );
	bool getMetadata( const std::string& name, std::string& description, 
			  boost::posix_time::ptime& insertionTime, boost::posix_time::ptime& modificationTime );
	void insert( const std::string& name, cond::TimeType timeType, const std::string& objectType, 
		     cond::SynchronizationType synchronizationType, cond::Time_t endOfValidity, const std::string& description, 
		     cond::Time_t lastValidatedTime, const boost::posix_time::ptime& insertionTime );
	void update( const std::string& name, cond::Time_t& endOfValidity, const std::string& description, 
		     cond::Time_t lastValidatedTime, const boost::posix_time::ptime& updateTime );
	void updateValidity( const std::string& name, cond::Time_t lastValidatedTime, const boost::posix_time::ptime& updateTime );
	void setValidationMode(){}
      private:
	coral::ISchema& m_schema;
      };
    }

    conddb_table ( PAYLOAD ) {
      
      static constexpr unsigned int PAYLOAD_HASH_SIZE = 40;
      
      conddb_column( HASH, std::string, PAYLOAD_HASH_SIZE );
      conddb_column( OBJECT_TYPE, std::string );
      conddb_column( DATA, cond::Binary );
      conddb_column( STREAMER_INFO, cond::Binary );
      conddb_column( VERSION, std::string );
      conddb_column( INSERTION_TIME, boost::posix_time::ptime );
     
      class Table : public IPayloadTable {
      public:
	explicit Table( coral::ISchema& schema );
	virtual ~Table(){}
	bool exists();
	void create();
	bool select( const cond::Hash& payloadHash);
	bool select( const cond::Hash& payloadHash, std::string& objectType, 
		     cond::Binary& payloadData, cond::Binary& streamerInfoData);
	bool getType( const cond::Hash& payloadHash, std::string& objectType );
	bool insert( const cond::Hash& payloadHash, const std::string& objectType, 
		     const cond::Binary& payloadData, const cond::Binary& streamerInfoData, 
		     const boost::posix_time::ptime& insertionTime);
	cond::Hash insertIfNew( const std::string& objectType, const cond::Binary& payloadData, 
				const cond::Binary& streamerInfoData, const boost::posix_time::ptime& insertionTime );
      private:
	coral::ISchema& m_schema;
      };
    }
    
    conddb_table( IOV ) {
      
      conddb_column( TAG_NAME, std::string );
      conddb_column( SINCE, cond::Time_t );
      conddb_column( PAYLOAD_HASH, std::string, PAYLOAD::PAYLOAD_HASH_SIZE );
      conddb_column( INSERTION_TIME, boost::posix_time::ptime );

      struct SINCE_GROUP {					 
	typedef cond::Time_t type;				   
	static constexpr size_t size = 0;
	static std::string tableName(){ return SINCE::tableName(); }	
	static std::string fullyQualifiedName(){ 
	  return "MIN("+SINCE::fullyQualifiedName()+")";	  
	} 
	static std::string group(){
	  std::string sgroupSize = boost::lexical_cast<std::string>( cond::time::SINCE_GROUP_SIZE);
	  return "CAST("+SINCE::fullyQualifiedName()+"/"+sgroupSize+" AS INT )*"+sgroupSize;
	}
      };
 
      struct SEQUENCE_SIZE {
	typedef unsigned int type;
	static constexpr size_t size = 0;
	static std::string tableName(){ return SINCE::tableName(); }
	static std::string fullyQualifiedName(){
	  return "COUNT(*)";
	}
      };
     
      class Table : public IIOVTable {
      public:
	explicit Table( coral::ISchema& schema );
	virtual ~Table(){}
	bool exists();
	void create();
	size_t selectGroups( const std::string& tag, std::vector<cond::Time_t>& groups );
	size_t selectSnapshotGroups( const std::string& tag, const boost::posix_time::ptime& snapshotTime, 
				     std::vector<cond::Time_t>& groups );
	size_t selectLatestByGroup( const std::string& tag, cond::Time_t lowerGroup, cond::Time_t upperGroup, 
				    std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs);
	size_t selectSnapshotByGroup( const std::string& tag, cond::Time_t lowerGroup, cond::Time_t upperGroup, 
				      const boost::posix_time::ptime& snapshotTime, 
				      std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs);
	size_t selectLatest( const std::string& tag, std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs);
	size_t selectSnapshot( const std::string& tag,
                               const boost::posix_time::ptime& snapshotTime,
                               std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs);
	bool getLastIov( const std::string& tag, cond::Time_t& since, cond::Hash& hash );
	bool getSnapshotLastIov( const std::string& tag, const boost::posix_time::ptime& snapshotTime, cond::Time_t& since, cond::Hash& hash );
	bool getSize( const std::string& tag, size_t& size );
        bool getSnapshotSize( const std::string& tag, const boost::posix_time::ptime& snapshotTime, size_t& size );
	void insertOne( const std::string& tag, cond::Time_t since, cond::Hash payloadHash, const boost::posix_time::ptime& insertTime);
	void insertMany( const std::string& tag, const std::vector<std::tuple<cond::Time_t,cond::Hash,boost::posix_time::ptime> >& iovs );
	void erase( const std::string& tag );
      private:
	coral::ISchema& m_schema;
      };
    }
    
    conddb_table( TAG_LOG ) {

      conddb_column( TAG_NAME, std::string );
      conddb_column( EVENT_TIME, boost::posix_time::ptime );
      conddb_column( USER_NAME, std::string );
      conddb_column( HOST_NAME, std::string );
      conddb_column( COMMAND, std::string );
      conddb_column( ACTION, std::string );
      conddb_column( USER_TEXT, std::string );

      class Table : public ITagLogTable {
      public:
	explicit Table( coral::ISchema& schema );
	virtual ~Table(){}
	bool exists();
        void create();
        void insert( const std::string& tag, const boost::posix_time::ptime& eventTime, const std::string& userName, const std::string& hostName, 
		     const std::string& command, const std::string& action, const std::string& userText );
      private:
	coral::ISchema& m_schema;
      };
    }
    
    class IOVSchema : public IIOVSchema {
    public: 
      explicit IOVSchema( coral::ISchema& schema );
      virtual ~IOVSchema(){}
      bool exists();
      bool create();
      ITagTable& tagTable();
      IIOVTable& iovTable();
      ITagLogTable& tagLogTable();
      IPayloadTable& payloadTable();
    private:
      TAG::Table m_tagTable;
      IOV::Table m_iovTable;
      TAG_LOG::Table m_tagLogTable;
      PAYLOAD::Table m_payloadTable;
    };
  }
}
#endif
