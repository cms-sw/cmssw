#ifndef INCLUDE_ORA_DATABASE_H
#define INCLUDE_ORA_DATABASE_H

#include "Object.h"
#include "OId.h"
#include "Version.h"
#include "Container.h"
#include "Configuration.h"
#include "DatabaseUtility.h"
#include "Exception.h"
//
#include <set>
#include <vector>
#include <memory>
#include <boost/shared_ptr.hpp>

namespace coral {
  class ISessionProxy;
}

namespace ora {

  class ConnectionPool;
  class Transaction;
  class DatabaseImpl;
  class SharedSession;

  class Database {
    public:

    // 
    static std::string nameForContainer( const std::type_info& typeInfo );
    //
    static std::string nameForContainer( const std::string& className );

    public:

    // 
    Database();

    //
    Database( const Database& rhs );
    
    // 
    Database(boost::shared_ptr<ConnectionPool>& connectionPool);
    
    /// 
    virtual ~Database();

    /// 
    Database& operator=( const Database& rhs );

    ///
    Configuration& configuration();

    ///
    ora::Version schemaVersion( bool userSchema=false );

    /// 
    bool connect( const std::string& connectionString, bool readOnly=false );
    
    /// 
    bool connect( const std::string& connectionString, const std::string& asRole, bool readOnly=false );

    /// 
    bool connect( boost::shared_ptr<coral::ISessionProxy>& coralSession, const std::string& connectionString, const std::string& schemaName="" );

    /// 
    void disconnect();

    /// 
    bool isConnected();

    /// 
    const std::string& connectionString();

    /// 
    Transaction& transaction();

    /// 
    bool exists();

    /// 
    bool create( std::string userSchemaVersion = std::string("") );

    /// 
    bool drop();

    ///
    void setAccessPermission( const std::string& principal, bool forWrite );
    
    /// 
    std::set< std::string > containers();

    ///
    template <typename T> Container createContainer( const std::string& name );
    
    ///
    template <typename T> Container createContainer();

    ///
    Container createContainer( const std::string& className, std::string name="" );

    /// 
    bool dropContainer( const std::string& name );

    ///
    bool lockContainer( const std::string& name );

    /// 
    Container containerHandle( const std::string& name );

    /// 
    Container containerHandle( int contId );

    /// 
    Object fetchItem(const OId& oid);

    ///
    template <typename T> boost::shared_ptr<T> fetch( const OId& oid);

    /// 
    OId insertItem( const std::string& containerName, const Object& data );

    ///
    template <typename T> OId insert( const std::string& containerName, const T& data);

    ///
    template <typename T> OId insert( const T& data);

    /// 
    void updateItem(const OId& oid, const Object& data );

    ///
    template <typename T> void update( const OId& oid, const T& data );

    /// 
    void erase(const OId& oid);

    ///
    void flush();

    ///
    void setObjectName( const std::string& name, const OId& oid );
   
    ///
    bool eraseObjectName( const std::string& name );

    /// 
    bool eraseAllNames();

    ///
    bool getItemId( const std::string& name, OId& destination );

    ///
    Object fetchItemByName( const std::string& name );

    ///
    template <typename T> boost::shared_ptr<T> fetchByName( const std::string& name );

    ///
    bool getNamesForObject( const OId& oid, std::vector<std::string>& destination );

    ///
    bool listObjectNames( std::vector<std::string>& destination );

    ///
    DatabaseUtility utility();

    public:

    ///
    SharedSession& storageAccessSession();

    private:

    ///
    void open( bool writingAccess=false );

    /// 
    void checkTransaction();

    /// 
    Container createContainer( const std::string& name, const std::type_info& typeInfo );
    
    ///  
    Container createContainer( const std::type_info& typeInfo );

    ///
    Container getContainer( const std::string& name, const std::type_info& typeInfo );

    ///
    Container getContainer( const std::type_info& typeInfo );

    ///
    boost::shared_ptr<void> getTypedObjectByName( const std::string& name, const std::type_info& typeInfo );    

    private:

    boost::shared_ptr<DatabaseImpl> m_impl;
    
  };

}

template <typename T>
inline
ora::Container ora::Database::createContainer( const std::string& name ){
  return createContainer( name, typeid(T) );
}

template <typename T>
inline
ora::Container ora::Database::createContainer(){
  return createContainer( typeid(T) );
}

template <typename T>
inline
boost::shared_ptr<T> ora::Database::fetch( const ora::OId& oid){
  Container cont = containerHandle( oid.containerId() );
  return cont.fetch<T>( oid.itemId() );
}

template <typename T>
inline
ora::OId ora::Database::insert(const std::string& containerName,
                               const T& data){
  Container cont = getContainer( containerName, typeid(data) );
  int itemId = cont.insert( data );
  return OId( cont.id(), itemId );
}

template <typename T>
inline
ora::OId ora::Database::insert( const T& data){
  Container cont = getContainer( typeid(data) );
  int itemId =  cont.insert( data );
  return OId( cont.id(), itemId );  
}

template <typename T>
inline
void ora::Database::update( const ora::OId& oid,
                            const T& data ){
  Container cont = containerHandle( oid.containerId() );
  cont.update( oid.itemId(), data );
}

template <typename T>
inline
boost::shared_ptr<T> ora::Database::fetchByName( const std::string& name ){
  return boost::static_pointer_cast<T>( getTypedObjectByName( name, typeid(T) ) );
}


#endif
