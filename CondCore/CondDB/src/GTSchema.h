#ifndef CondCore_CondDB_GTSchema_h
#define CondCore_CondDB_GTSchema_h

#include "DbCore.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {

    class SessionImpl;
    
    table( GLOBAL_TAG ) {
      
      column( NAME, std::string );
      column( VALIDITY, cond::Time_t );
      column( DESCRIPTION, std::string );
      column( RELEASE, std::string );
      column( SNAPSHOT_TIME, boost::posix_time::ptime );
      column( INSERTION_TIME, boost::posix_time::ptime );
      
      bool exists( SessionImpl& session );
      bool select( const std::string& name, SessionImpl& session );
      bool select( const std::string& name, cond::Time_t& validity, boost::posix_time::ptime& snapshotTime, SessionImpl& session );
      bool select( const std::string& name, cond::Time_t& validity, std::string& description, 
		   std::string& release, boost::posix_time::ptime& snapshotTime, SessionImpl& session );
      void insert( const std::string& name, cond::Time_t validity, const std::string& description, const std::string& release, 
		   const boost::posix_time::ptime& snapshotTime, const boost::posix_time::ptime& insertionTime, SessionImpl& session );
      void update( const std::string& name, cond::Time_t validity, const std::string& description, const std::string& release, 
		   const boost::posix_time::ptime& snapshotTime, const boost::posix_time::ptime& insertionTime, SessionImpl& session );
    }
    
    table ( GLOBAL_TAG_MAP ) {
      
      static constexpr unsigned int PAYLOAD_HASH_SIZE = 40;
      
      column( GLOBAL_TAG_NAME, std::string );
      // to be changed to RECORD_NAME!
      column( RECORD, std::string );
      // to be changed to RECORD_LABEL!
      column( LABEL, std::string );
      column( TAG_NAME, std::string );
      
      bool exists( SessionImpl& session );
      bool select( const std::string& gtName, std::vector<std::tuple<std::string,std::string,std::string> >& tags, SessionImpl& session );
      void insert( const std::string& gtName, const std::vector<std::tuple<std::string,std::string,std::string> >& tags, SessionImpl& session );
    }
    
    namespace gtDb {
      bool exists( SessionImpl& session );
    }
    
  }
}
#endif
