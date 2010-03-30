#ifndef INCLUDE_ORA_DATABASESESSION_H
#define INCLUDE_ORA_DATABASESESSION_H

#include "CondCore/ORA/interface/Handle.h"
#include "CondCore/ORA/interface/ConnectionPool.h"
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
    explicit DatabaseSession( Configuration& configuration );

    DatabaseSession(boost::shared_ptr<ConnectionPool>& connectionPool, Configuration& configuration );

    virtual ~DatabaseSession();

    bool connect( const std::string& connectionString, bool readOnly );

    void disconnect();

    bool isConnected();

    const std::string& connectionString();

    void startTransaction( bool readOnly );

    void commitTransaction();

    void rollbackTransaction();

    bool isTransactionActive();
    
    bool exists();

    void create();

    void drop();

    void open();

    Handle<DatabaseContainer> createContainer( const std::string& containerName, const Reflex::Type& type );
    
    void dropContainer( const std::string& name );

    Handle<DatabaseContainer> containerHandle( const std::string& name );

    Handle<DatabaseContainer> containerHandle( int contId );

    const std::map<int, Handle<DatabaseContainer> >& containers();

    public:
    IDatabaseSchema& schema();

    NamedSequence& containerIdSequence();

    MappingDatabase& mappingDatabase();

    ContainerUpdateTable& containerUpdateTable();

    Configuration& configuration();
    
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
    Configuration& m_configuration;
  };
  

}

#endif
