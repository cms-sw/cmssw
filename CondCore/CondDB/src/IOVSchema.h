#ifndef CondCore_CondDB_IOVSchema_h
#define CondCore_CondDB_IOVSchema_h

#include "DbCore.h"
#include "IDbSchema.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {

    table( TAG ) {
      
      column( NAME, std::string );
      column( TIME_TYPE, cond::TimeType );
      column( OBJECT_TYPE, std::string );
      column( SYNCHRONIZATION, cond::SynchronizationType );
      column( END_OF_VALIDITY, cond::Time_t );
      column( DESCRIPTION, std::string );
      column( LAST_VALIDATED_TIME, cond::Time_t );
      column( INSERTION_TIME, boost::posix_time::ptime );
      column( MODIFICATION_TIME, boost::posix_time::ptime );
      
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

    table ( PAYLOAD ) {
      
      static constexpr unsigned int PAYLOAD_HASH_SIZE = 40;
      
      column( HASH, std::string, PAYLOAD_HASH_SIZE );
      column( OBJECT_TYPE, std::string );
      column( DATA, cond::Binary );
      column( STREAMER_INFO, cond::Binary );
      column( VERSION, std::string );
      column( INSERTION_TIME, boost::posix_time::ptime );
     
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
    
    table( IOV ) {
      
      column( TAG_NAME, std::string );
      column( SINCE, cond::Time_t );
      column( PAYLOAD_HASH, std::string, PAYLOAD::PAYLOAD_HASH_SIZE );
      column( INSERTION_TIME, boost::posix_time::ptime );
      
      struct MAX_SINCE {					 
	typedef cond::Time_t type;				   
	static constexpr size_t size = 0;
	static std::string tableName(){ return SINCE::tableName(); }	
	static std::string fullyQualifiedName(){ 
	  return std::string("MAX(")+SINCE::fullyQualifiedName()+")";
	} 
      };
      struct SINCE_GROUP {					 
	typedef cond::Time_t type;				   
	static constexpr size_t size = 0;
	static std::string tableName(){ return SINCE::tableName(); }	
	static std::string fullyQualifiedName(){ 
	  std::string sgroupSize = boost::lexical_cast<std::string>(cond::time::SINCE_GROUP_SIZE);
	  return "("+SINCE::fullyQualifiedName()+"/"+sgroupSize+")*"+sgroupSize;	  
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
	bool getLastIov( const std::string& tag, cond::Time_t& since, cond::Hash& hash );
	bool getSize( const std::string& tag, size_t& size );
        bool getSnapshotSize( const std::string& tag, const boost::posix_time::ptime& snapshotTime, size_t& size );
	void insertOne( const std::string& tag, cond::Time_t since, cond::Hash payloadHash, const boost::posix_time::ptime& insertTime);
	void insertMany( const std::string& tag, const std::vector<std::tuple<cond::Time_t,cond::Hash,boost::posix_time::ptime> >& iovs );
      private:
	coral::ISchema& m_schema;
      };
    }
    
    // temporary... to be removed after the changeover.
    table( TAG_MIGRATION ) {
      
      column( SOURCE_ACCOUNT, std::string );
      column( SOURCE_TAG, std::string );
      column( TAG_NAME, std::string );
      column( INSERTION_TIME, boost::posix_time::ptime );
      
      class Table : public ITagMigrationTable {
      public:
	explicit Table( coral::ISchema& schema );
	virtual ~Table(){}
	bool exists();
	void create();
	bool select( const std::string& sourceAccount, const std::string& sourceTag, std::string& tagName);
	void insert( const std::string& sourceAccount, const std::string& sourceTag, const std::string& tagName, 
		     const boost::posix_time::ptime& insertionTime);
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
      IPayloadTable& payloadTable();
      ITagMigrationTable& tagMigrationTable();
    private:
      TAG::Table m_tagTable;
      IOV::Table m_iovTable;
      PAYLOAD::Table m_payloadTable;
      TAG_MIGRATION::Table m_tagMigrationTable;
    };
  }
}
#endif
