#ifndef INCLUDE_ORA_DATABASESESSION_H
#define INCLUDE_ORA_DATABASESESSION_H

#include "CondCore/ORA/interface/Handle.h"
#include "CondCore/ORA/interface/Object.h"
#include "CondCore/ORA/interface/ConnectionPool.h"
#include "CondCore/ORA/interface/Configuration.h"
//
#include <string>
#include <memory>
#include <map>

namespace Reflex {
  class Type;
}

namespace ora {

  class Configuration;
  class ConnectionPool;
  class NamedSequence;
  class MappingDatabase;
  class IDatabaseSchema;
  class TransactionCache;
  class DatabaseContainer;
  class DatabaseUtilitySession;

  class ContainerUpdateTable {
    public:
    ContainerUpdateTable();
    ~ContainerUpdateTable();
    void takeNote( int contId, unsigned int size );
    const std::map<int, unsigned int>& table();
    void clear();
    private:
    std::map<int, unsigned int> m_table;
  };

  class DatabaseSession  {
    
    public:
    DatabaseSession();

    explicit DatabaseSession( boost::shared_ptr<ConnectionPool>& connectionPool );

    virtual ~DatabaseSession();

    bool connect( const std::string& connectionString, bool readOnly );

    void disconnect();

    bool isConnected();

    const std::string& connectionString();

    void startTransaction( bool readOnly );

    void commitTransaction();

    void rollbackTransaction();

    bool isTransactionActive( bool checkIfReadOnly=false );
    
    bool exists();

    void create();

    void drop();

    void open();

    Handle<DatabaseContainer> createContainer( const std::string& containerName, const Reflex::Type& type );

    Handle<ora::DatabaseContainer> addContainer( const std::string& containerName, const std::string& className );
    
    void dropContainer( const std::string& name );

    Handle<DatabaseContainer> containerHandle( const std::string& name );

    Handle<DatabaseContainer> containerHandle( int contId );

    const std::map<int, Handle<DatabaseContainer> >& containers();

    void setObjectName( const std::string& name, int containerId, int itemId );
    bool eraseObjectName( const std::string& name );
    ora::Object fetchObjectByName( const std::string& name );
    boost::shared_ptr<void> fetchTypedObjectByName( const std::string& name, const Reflex::Type& asType );
    bool getNamesForContainer( int containerId, std::vector<std::string>& destination );
    bool getNamesForObject( int containerId, int itemId, std::vector<std::string>& destination );

    Handle<DatabaseUtilitySession> utility();

    public:
    IDatabaseSchema& schema();

    NamedSequence& containerIdSequence();

    MappingDatabase& mappingDatabase();

    ContainerUpdateTable& containerUpdateTable();

    Configuration& configuration();
    
    public:
    SharedSession& storageAccessSession();
    boost::shared_ptr<ConnectionPool>& connectionPool();
    
    private:
    void clearTransaction();

    private:
    boost::shared_ptr<ConnectionPool> m_connectionPool;
    SharedSession m_dbSession;
    std::string m_connectionString;
    std::auto_ptr<IDatabaseSchema> m_schema;
    std::auto_ptr<NamedSequence> m_contIdSequence;
    std::auto_ptr<MappingDatabase> m_mappingDb;
    std::auto_ptr<TransactionCache> m_transactionCache;
    ContainerUpdateTable m_containerUpdateTable;
    Configuration m_configuration;
  };
  

}

#endif
