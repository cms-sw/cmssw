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

namespace cond{

  class DbTransaction;
  class DbConnection;
  class SessionImpl;

  /*
  **/
  class DbSession{
  public:
  static const char* COND_SCHEMA_VERSION;   
  static const char* CHANGE_SCHEMA_VERSION;

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

    bool createDatabase();

    // TEMPORARY, for the IOV schema changeover.
    bool isOldSchema();

    coral::ISchema& schema(const std::string& schemaName);

    coral::ISchema& nominalSchema();

    bool deleteMapping( const std::string& mappingVersion );

    bool importMapping( const std::string& sourceConnectionString,
                        const std::string& contName );

    ora::Object getObject( const std::string& objectId );

    template <typename T> boost::shared_ptr<T> getTypedObject( const std::string& objectId );

    template <typename T> std::string storeObject( const T* object, const std::string& containerName );

    template <typename T> bool updateObject( const T* object, const std::string& objectId );

    bool deleteObject( const std::string& objectId );

    std::string importObject( cond::DbSession& fromDatabase, const std::string& objectId );

    std::string classNameForItem( const std::string& objectId );

    void flush();
    
    ora::Database& storage();
    
    private:
    std::string storeObject( const ora::Object& objectRef, const std::string& containerName  );
    private:

    boost::shared_ptr<SessionImpl> m_implementation;
  };

  class PoolTokenParser : public ora::ITokenParser {
    public:
    explicit PoolTokenParser( ora::Database& db );
    ~PoolTokenParser(){
    }
    ora::OId parse( const std::string& poolToken );
    std::string className( const std::string& poolToken );

    private:
    ora::Database& m_db;
  };

  class PoolTokenWriter : public ora::ITokenWriter {
    public:
    explicit PoolTokenWriter( ora::Database& db );
    ~PoolTokenWriter(){
    }
    std::string write( const ora::OId& oid );
    private:
    ora::Database& m_db;
  };

  template <typename T> inline boost::shared_ptr<T> DbSession::getTypedObject( const std::string& objectId ){
    ora::OId oid;
    oid.fromString( objectId );
    return storage().fetch<T>( oid );
  }

  template <typename T> inline std::string DbSession::storeObject( const T* object,
                                                                   const std::string& containerName ){
    std::string ret("");
    if( object ){
      ora::OId oid = storage().insert( containerName, *object );
      storage().flush();
      ret =  oid.toString();
    }
    return ret;
  }

  template <typename T> inline bool DbSession::updateObject( const T* object,
                                                             const std::string& objectId ){
    bool ret = false;
    if( object ){
      ora::OId oid;
      oid.fromString( objectId );
      storage().update( oid, *object );
      storage().flush();
      ret =  true;
    }
    return ret;
  }
 
}

#endif
// DBSESSION_H
