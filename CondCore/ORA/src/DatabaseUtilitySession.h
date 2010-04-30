#ifndef INCLUDE_ORA_DATABASEUTILITYIMPL_H
#define INCLUDE_ORA_DATABASEUTILITYIMPL_H

#include "CondCore/ORA/interface/Handle.h"
//
#include <set>

namespace ora {

  class DatabaseSession;
  class DatabaseContainer;
  
  class DatabaseUtilitySession {
    public:
    explicit DatabaseUtilitySession( DatabaseSession& dbSession );

    virtual ~DatabaseUtilitySession();

    std::set<std::string> listMappingVersions( const std::string& containerName );

    void importContainerSchema( const std::string& sourceConnectionString, const std::string& containerName );

    void importContainer( const std::string& sourceConnectionString, const std::string& containerName );

    void eraseMapping( const std::string& mappingVersion );

    private:

    Handle<ora::DatabaseContainer> importContainerSchema( const std::string& containerName, DatabaseSession& sourceDbSession );

    bool existsContainer( const std::string& containerName );

    private:

    DatabaseSession& m_session;
  };
}

#endif
 
