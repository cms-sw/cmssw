#ifndef CondCore_CondDB_IOVSchema_h
#define CondCore_CondDB_IOVSchema_h

#include "DbCore.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>

//namespace coral {
//  class ISchema;
//}

namespace cond {

  namespace persistency {

    class SessionImpl;

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
      
      bool exists( SessionImpl& session );
      void create( SessionImpl& session );
      bool select( const std::string& name, SessionImpl& session );
      bool select( const std::string& name, cond::TimeType& timeType, std::string& objectType, cond::Time_t& endOfValidity, std::string& description, cond::Time_t& lastValidatedTime, SessionImpl& session );
      bool getMetadata( const std::string& name, std::string& description, boost::posix_time::ptime& insertionTime, boost::posix_time::ptime& modificationTime, SessionImpl& session );
      void insert( const std::string& name, cond::TimeType timeType, const std::string& objectType, cond::SynchronizationType synchronizationType, 
		   cond::Time_t endOfValidity, const std::string& description, cond::Time_t lastValidatedTime, const boost::posix_time::ptime& insertionTime, SessionImpl& session  );
      void update( const std::string& name, cond::Time_t& endOfValidity, const std::string& description, cond::Time_t lastValidatedTime, 
		   const boost::posix_time::ptime& updateTime, SessionImpl& session );
      void updateValidity( const std::string& name, cond::Time_t lastValidatedTime, const boost::posix_time::ptime& updateTime, SessionImpl& session );
    }

    table ( PAYLOAD ) {
      
      static constexpr unsigned int PAYLOAD_HASH_SIZE = 40;
      
      column( HASH, std::string, PAYLOAD_HASH_SIZE );
      column( OBJECT_TYPE, std::string );
      column( DATA, cond::Binary );
      //column( STREAMER, std::string );
      column( STREAMER_INFO, cond::Binary );
      column( VERSION, std::string );
      column( INSERTION_TIME, boost::posix_time::ptime );
      
      bool exists( SessionImpl& session );
      void create( SessionImpl& session );
      bool select( const cond::Hash& payloadHash, SessionImpl& session );
      bool select( const cond::Hash& payloadHash, std::string& objectType, cond::Binary& payloadData, SessionImpl& session );
      bool insert( const cond::Hash& payloadHash, const std::string& objectType, const cond::Binary& payloadData, const boost::posix_time::ptime& insertionTime, SessionImpl& session );
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
      
      bool exists( SessionImpl& session );
      void create( SessionImpl& session );
      size_t selectGroups( const std::string& tag, std::vector<cond::Time_t>& groups, SessionImpl& session );
      size_t selectSnapshotGroups( const std::string& tag, const boost::posix_time::ptime& snapshotTime, std::vector<cond::Time_t>& groups,SessionImpl& session );
      size_t selectLastByGroup( const std::string& tag, cond::Time_t lowerGroup, cond::Time_t upperGroup , 
				std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs, 
				SessionImpl& session );
      size_t selectSnapshotByGroup( const std::string& tag, cond::Time_t lowerGroup, cond::Time_t upperGroup, 
				    const boost::posix_time::ptime& snapshotTime, 
				    std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs, 
				    SessionImpl& session );
      //size_t selectLastByGroup( const std::string& tag, cond::Time_t target, 
      //			      std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs, SessionImpl& session );
      //size_t selectSnapshotByGroup( const std::string& tag, cond::Time_t target, const boost::posix_time::ptime& snapshotUpperTime, 
      //				  std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs, SessionImpl& session );
      size_t selectLast( const std::string& tag, std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs, SessionImpl& session );
      void insertOne( const std::string& tag, cond::Time_t since, cond::Hash payloadHash, const boost::posix_time::ptime& insertTime, SessionImpl& session );
      void insertMany( const std::string& tag, const std::vector<std::tuple<cond::Time_t,cond::Hash,boost::posix_time::ptime> >& iovs, SessionImpl& session );
    }
    
    // temporary... to be removed after the changeover.
    table( TAG_MIGRATION ) {
      
      column( SOURCE_ACCOUNT, std::string );
      column( SOURCE_TAG, std::string );
      column( TAG_NAME, std::string );
      column( INSERTION_TIME, boost::posix_time::ptime );
      
      bool exists( SessionImpl& session );
      void create( SessionImpl& session );
      bool select( const std::string& sourceAccount, const std::string& sourceTag, std::string& tagName, SessionImpl& session );
      void insert( const std::string& sourceAccount, const std::string& sourceTag, const std::string& tagName, 
		   const boost::posix_time::ptime& insertionTime, SessionImpl& session  );
    }
    
    namespace iovDb {
      bool exists( SessionImpl& session );
      bool create( SessionImpl& session );
    }
  }
}
#endif
