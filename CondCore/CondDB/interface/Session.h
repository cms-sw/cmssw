#ifndef CondCore_CondDB_Session_h
#define CondCore_CondDB_Session_h
//
// Package:     CondDB
// Class  :     Session
// 
/**\class Session Session.h CondCore/CondDB/interface/Session.h
   Description: service for accessing conditions in/from DB.  
*/
//
// Author:      Giacomo Govi
// Created:     Apr 2013
//

#include "CondCore/CondDB/interface/IOVProxy.h"
#include "CondCore/CondDB/interface/IOVEditor.h"
#include "CondCore/CondDB/interface/GTProxy.h"
#include "CondCore/CondDB/interface/GTEditor.h"
#include "CondCore/CondDB/interface/RunInfoProxy.h"
#include "CondCore/CondDB/interface/RunInfoEditor.h"
#include "CondCore/CondDB/interface/Binary.h"
#include "CondCore/CondDB/interface/Serialization.h"
#include "CondCore/CondDB/interface/Types.h"
#include "CondCore/CondDB/interface/Utils.h"
// 
//#include <vector>
//#include <tuple>
// temporarely

// TO BE REMOVED AFTER THE TRANSITION
namespace coral {
  class ISessionProxy;
  class ISchema;
}
// END TO BE REMOVED

namespace cond {

  namespace persistency {

    class SessionConfiguration; 

    // transaction class 
    class Transaction {
    public:
      explicit Transaction( SessionImpl& session );
      Transaction( const Transaction& rhs );
      Transaction& operator=( const Transaction& rhs );
      
      void start( bool readOnly=true );
      
      void commit();
      
      void rollback();
      
      bool isActive(); 
    private:
      SessionImpl* m_session;
    };
    
    // with full value semantics. destruction will close the DB connection.
    class Session {
    public:
      // default constructor
      Session();
      
      // constructor
      explicit Session( const std::shared_ptr<SessionImpl>& sessionImpl );

      // 
      Session( const Session& rhs );
      
      // 
      virtual ~Session();
      
      //
      Session& operator=( const Session& rhs );
      
      // 
      void close();
      
      //
      Transaction& transaction();
      
      //
      bool existsDatabase();

      //
      void createDatabase();
      
      // read access to the iov sequence. 
      // the iovs are lazy-loaded in groups when required, with repeatable queries ( for FronTier )
      IOVProxy readIov( const std::string& tag, bool full=false );

      // read access to the iov sequence. 
      // the iovs are lazy-loaded in groups when required, with repeatable queries ( for FronTier )
      IOVProxy readIov( const std::string& tag, 
			const boost::posix_time::ptime& snapshottime,
			bool full=false );  
      
      // 
      bool existsIov( const std::string& tag );
      
      // retrieves an IOV range. Peforms a query at every call.
      bool getIovRange( const std::string& tag, 
			cond::Time_t begin, cond::Time_t end, 
			std::vector<std::tuple<cond::Time_t,cond::Hash> >& range );

      // create a non-existing iov sequence with the specified tag.
      // the type is required for consistency with the referenced payloads.    
      // fixme: add creation time - required for the migration
      template <typename T>
      IOVEditor createIov( const std::string& tag, 
			   cond::TimeType timeType, 
			   cond::SynchronizationType synchronizationType=cond::SYNCH_ANY );
      IOVEditor createIov( const std::string& payloadType, 
			   const std::string& tag, 
			   cond::TimeType timeType,
			   cond::SynchronizationType synchronizationType=cond::SYNCH_ANY );

      IOVEditor createIov( const std::string& payloadType, 
			   const std::string& tag, 
			   cond::TimeType timeType,
			   cond::SynchronizationType synchronizationType,
			   const boost::posix_time::ptime& creationTime );

      IOVEditor createIovForPayload( const Hash& payloadHash, 
				     const std::string& tag, cond::TimeType timeType,
				     cond::SynchronizationType synchronizationType=cond::SYNCH_ANY );

      void clearIov( const std::string& tag );
      
      // update an existing iov sequence with the specified tag.
      // timeType and payloadType can't be modified.
      IOVEditor editIov( const std::string& tag );

      // functions to store a payload in the database. return the identifier of the item in the db. 
      template <typename T> cond::Hash storePayload( const T& payload, 
						     const boost::posix_time::ptime& creationTime = boost::posix_time::microsec_clock::universal_time() );

      template <typename T> std::unique_ptr<T> fetchPayload( const cond::Hash& payloadHash );
      
      cond::Hash storePayloadData( const std::string& payloadObjectType,
                                   const std::pair<Binary,Binary>& payloadAndStreamerInfoData,
                                   const boost::posix_time::ptime& creationTime );

      bool fetchPayloadData( const cond::Hash& payloadHash, 
			     std::string& payloadType, 
			     cond::Binary& payloadData,
			     cond::Binary& streamerInfoData );

      // internal functions. creates proxies without loading a specific tag.  
      IOVProxy iovProxy();
      
      bool existsGlobalTag( const std::string& name );

      GTEditor createGlobalTag( const std::string& name );
      GTEditor editGlobalTag( const std::string& name );
      
      GTProxy readGlobalTag( const std::string& name );
      // essentially for the bridge. useless where ORA disappears.
      GTProxy readGlobalTag( const std::string& name, 
			     const std::string& preFix, 
			     const std::string& postFix  );

      // runinfo read only access
      RunInfoProxy getRunInfo( cond::Time_t start, cond::Time_t end );

      // runinfo write access
      RunInfoEditor editRunInfo();
    public:
      
      std::string connectionString();

      coral::ISessionProxy& coralSession();
      // TO BE REMOVED in the long term. The new code will use coralSession().
      coral::ISchema& nominalSchema();
      
    private:
      
      std::shared_ptr<SessionImpl> m_session;
      Transaction m_transaction;
    };
    
    template <typename T> inline IOVEditor Session::createIov( const std::string& tag, cond::TimeType timeType, cond::SynchronizationType synchronizationType ){
      return createIov( cond::demangledName( typeid(T) ), tag, timeType, synchronizationType );
    }
    
    template <typename T> inline cond::Hash Session::storePayload( const T& payload, const boost::posix_time::ptime& creationTime ){
      
      std::string payloadObjectType = cond::demangledName(typeid(payload));
      cond::Hash ret; 
      try{
	ret = storePayloadData( payloadObjectType, serialize( payload ), creationTime ); 
      } catch ( const cond::persistency::Exception& e ){
	std::string em(e.what());
	throwException( "Payload of type "+payloadObjectType+" could not be stored. "+em,"Session::storePayload"); 	
      }
      return ret;
    }

    template <> inline cond::Hash Session::storePayload<std::string>( const std::string& payload, const boost::posix_time::ptime& creationTime ){

      std::string payloadObjectType("std::string");
      cond::Hash ret;
      try{
        ret = storePayloadData( payloadObjectType, serialize( payload ), creationTime );
      } catch ( const cond::persistency::Exception& e ){
	std::string em(e.what());
        throwException( "Payload of type "+payloadObjectType+" could not be stored. "+em,"Session::storePayload");
      }
      return ret;
    }
    
    template <typename T> inline std::unique_ptr<T> Session::fetchPayload( const cond::Hash& payloadHash ){
      cond::Binary payloadData;
      cond::Binary streamerInfoData;
      std::string payloadType;
      if(! fetchPayloadData( payloadHash, payloadType, payloadData, streamerInfoData ) ) 
	throwException( "Payload with id "+payloadHash+" has not been found in the database.",
			"Session::fetchPayload" );
      std::unique_ptr<T> ret;
      try{ 
	ret = deserialize<T>(  payloadType, payloadData, streamerInfoData );
      } catch ( const cond::persistency::Exception& e ){
	std::string em(e.what());
	throwException( "Payload of type "+payloadType+" with id "+payloadHash+" could not be loaded. "+em,"Session::fetchPayload"); 
      }
      return ret;
    }

    class TransactionScope {
    public:
      explicit TransactionScope( Transaction& transaction );   
      
      ~TransactionScope();

      void start( bool readOnly=true );

      void commit();
      
      void close();
    private:
      Transaction& m_transaction;
      bool m_status;
      
    };


  }
}
#endif
