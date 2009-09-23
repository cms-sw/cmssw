#ifndef COND_DBCommon_DbSession_h
#define COND_DBCommon_DbSession_h

#include <string>
#include <boost/shared_ptr.hpp>
#include "DataSvc/Ref.h"


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

    void setBlobStreamingService( const std::string& streamingService );
    
    const std::string& blobStreamingService() const;

    void open( const std::string& connectionString, bool readOnly=false );

    void close();

    bool isOpen() const;

    DbTransaction& transaction();

    coral::ISchema& schema(const std::string& schemaName);

    coral::ISchema& nominalSchema();

    coral::ISessionProxy& coralSession();

    pool::IDataSvc& poolCache();

    bool initializeMapping(const std::string& mappingVersion, const std::string& xmlStream);

    bool deleteMapping( const std::string& mappingVersion, bool removeTables );

    bool importMapping( cond::DbSession& fromDatabase,
                        const std::string& contName,
                        const std::string& classVersion="",
                        bool allVersions=false );

    pool::RefBase getObject( const std::string& objectId );

    template <typename T> pool::Ref<T> getTypedObject( const std::string& objectId );

    template <typename T> pool::Ref<T> storeObject( T* object, const std::string& containerName );

    bool deleteObject( const std::string& objectId );

    std::string importObject( cond::DbSession& fromDatabase, const std::string& objectId );

    private:
    class SessionImpl;

    private:

    bool storeObject( pool::RefBase& objectRef, const std::string& containerName  );

    private:

    boost::shared_ptr<SessionImpl> m_implementation;
  };

template <typename T> inline pool::Ref<T> DbSession::getTypedObject( const std::string& objectId ){
  return pool::Ref<T>(&poolCache(),objectId );
}

template <typename T> inline pool::Ref<T> DbSession::storeObject( T* object, const std::string& containerName ){
  pool::Ref<T> objectRef(&poolCache(), object );
  storeObject( objectRef, containerName );
  return objectRef;
}
 
}

#endif
// DBSESSION_H
