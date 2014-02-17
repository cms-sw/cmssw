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
#include "CondCore/CondDB/interface/Binary.h"
#include "CondCore/CondDB/interface/Serialization.h"
#include "CondCore/CondDB/interface/Types.h"
#include "CondCore/CondDB/interface/Utils.h"
// 
// temporarely
#include <boost/shared_ptr.hpp>

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
      // by default ( full=false ) the iovs are lazy-loaded in groups when required, with repeatable queries ( for FronTier )
      // full=true will load the entire sequence in memory. Mainly for test/debugging.
      IOVProxy readIov( const std::string& tag, bool full=false );//,const boost::posix_time::ptime& snapshottime )  
      
      // 
      bool existsIov( const std::string& tag );
      
      // create a non-existing iov sequence with the specified tag.
      // the type is required for consistency with the referenced payloads.    
      // fixme: add creation time - required for the migration
      template <typename T>
      IOVEditor createIov( const std::string& tag, cond::TimeType timeType, 
			   cond::SynchronizationType synchronizationType=cond::OFFLINE );
      IOVEditor createIov(  const std::string& payloadType, const std::string& tag, cond::TimeType timeType,
			   cond::SynchronizationType synchronizationType=cond::OFFLINE );

      IOVEditor createIovForPayload(  const Hash& payloadHash, const std::string& tag, cond::TimeType timeType,
				      cond::SynchronizationType synchronizationType=cond::OFFLINE );
      
      // update an existing iov sequence with the specified tag.
      // timeType and payloadType can't be modified.
      IOVEditor editIov( const std::string& tag );
      
      // functions to store a payload in the database. return the identifier of the item in the db. 
      template <typename T> cond::Hash storePayload( const T& payload, 
						     const boost::posix_time::ptime& creationTime = boost::posix_time::microsec_clock::universal_time() );
      template <typename T> boost::shared_ptr<T> fetchPayload( const cond::Hash& payloadHash );
      
      // low-level function to access the payload data as a blob. mainly used for the data migration and testing. 
      bool fetchPayloadData( const cond::Hash& payloadHash, std::string& payloadType, cond::Binary& payloadData );
      
      // internal functions. creates proxies without loading a specific tag.  
      IOVProxy iovProxy();
      
      GTEditor createGlobalTag( const std::string& name );
      GTEditor editGlobalTag( const std::string& name );
      
      GTProxy readGlobalTag( const std::string& name );
      // essentially for the bridge. useless where ORA disappears.
      GTProxy readGlobalTag( const std::string& name, const std::string& preFix, const std::string& postFix  );
    public:
      
      bool checkMigrationLog( const std::string& sourceAccount, const std::string& sourceTag, std::string& destinationTag );
      void addToMigrationLog( const std::string& sourceAccount, const std::string& sourceTag, const std::string& destinationTag );

      std::string connectionString();

      coral::ISessionProxy& coralSession();
      // TO BE REMOVED in the long term. The new code will use coralSession().
      coral::ISchema& nominalSchema();
      
      bool isOraSession(); 

    private:
      cond::Hash storePayloadData( const std::string& payloadObjectType, const cond::Binary& payloadData, const boost::posix_time::ptime& creationTime );
      
    private:
      
      std::shared_ptr<SessionImpl> m_session;
      Transaction m_transaction;
    };
    
    template <typename T> inline IOVEditor Session::createIov( const std::string& tag, cond::TimeType timeType, cond::SynchronizationType synchronizationType ){
      return createIov( cond::demangledName( typeid(T) ), tag, timeType, synchronizationType );
    }
    
    template <typename T> inline cond::Hash Session::storePayload( const T& payload, const boost::posix_time::ptime& creationTime ){
      
      std::string payloadObjectType = cond::demangledName(typeid(payload));
      return storePayloadData( payloadObjectType, serialize( payload, isOraSession() ), creationTime ); 
    }
    
    template <typename T> inline boost::shared_ptr<T> Session::fetchPayload( const cond::Hash& payloadHash ){
      cond::Binary payloadData;
      std::string payloadType;
      if(! fetchPayloadData( payloadHash, payloadType, payloadData ) ) 
	throwException( "Payload with id="+payloadHash+" has not been found in the database.",
			"Session::fetchPayload" );
      return deserialize<T>(  payloadType, payloadData, isOraSession() );
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
