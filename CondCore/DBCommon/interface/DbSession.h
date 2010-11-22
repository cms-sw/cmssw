#ifndef COND_DBCommon_DbSession_h
#define COND_DBCommon_DbSession_h

#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/PoolToken.h"
#include <string>
#include <boost/shared_ptr.hpp>


//
// Package:    CondCore/DBCommon
// Class:      DbSession
//
/**\class DbSession DbSession.h CondCore/DBCommon/interface/DbSession.h
 Description: Class to prepare database connection setup
*/

namespace coral {
  class IConnectionService;
  class ISchema;
  class ISessionProxy;
}

namespace pool {
  class IDataSvc;
}

namespace cond{

  class DbTransaction;
  class DbConnection;
  class SessionImpl;
  
  /*
  **/
  class DbSession{
  public:
    DbSession();

    explicit DbSession( const DbConnection& connection );

    DbSession( const DbSession& rhs );

    virtual ~DbSession();

    DbSession& operator=( const DbSession& rhs );

    const std::string& connectionString() const;

    const DbConnection& connection() const;

    bool isTransactional() const;

    const std::string& blobStreamingService() const;

    void open( const std::string& connectionString, bool readOnly=false );

    void close();

    bool isOpen() const;

    DbTransaction& transaction();

    coral::ISchema& schema(const std::string& schemaName);

    coral::ISchema& nominalSchema();

    //bool initializeMapping(const std::string& mappingVersion, const std::string& xmlStream);

    bool deleteMapping( const std::string& mappingVersion );

    bool importMapping( const std::string& sourceConnectionString,
                        const std::string& contName );

    ora::Object getObject( const std::string& objectId );

    template <typename T> boost::shared_ptr<T> getTypedObject( const std::string& objectId );

    template <typename T> std::string storeObject( const T* object, const std::string& containerName );

    template <typename T> bool updateObject( const T* object, const std::string& objectId );

    bool deleteObject( const std::string& objectId );

    std::string importObject( cond::DbSession& fromDatabase, const std::string& objectId );

    void flush();
    
    ora::Database& storage();
    
    private:
    std::string storeObject( const ora::Object& objectRef, const std::string& containerName  );
    private:

    boost::shared_ptr<SessionImpl> m_implementation;
  };

  template <typename T> inline boost::shared_ptr<T> DbSession::getTypedObject( const std::string& objectId ){
    std::pair<std::string,int> oidData = parseToken( objectId );
    ora::Container cont = storage().containerHandle(  oidData.first );
    return cont.fetch<T>( oidData.second );
  }

  template <typename T> inline std::string DbSession::storeObject( const T* object,
                                                                   const std::string& containerName ){
    std::string ret("");
    if( object ){
      ora::OId oid = storage().insert( containerName, *object );
      storage().flush();
      ora::Container cont = storage().containerHandle( containerName );
      int oid0 = cont.id(); // contID does not start from 0...
      ret =  writeToken( containerName, oid0, oid.itemId(), cont.className() );
    }
    return ret;
  }

  template <typename T> inline bool DbSession::updateObject( const T* object,
                                                             const std::string& objectId ){
    bool ret = false;
    if( object ){
      std::pair<std::string,int> oidData = parseToken( objectId );
      ora::Container cont = storage().containerHandle( oidData.first );
      cont.update( oidData.second, *object );
      cont.flush();
      ret =  true;
    }
    return ret;
  }
 
}

#endif
// DBSESSION_H
