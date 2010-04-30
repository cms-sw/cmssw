#ifndef INCLUDE_ORA_DATABASEUTILITY_H
#define INCLUDE_ORA_DATABASEUTILITY_H

#include "Handle.h"
#include <boost/shared_ptr.hpp>
//
#include <set>

namespace ora {

  class DatabaseUtilitySession;
  
  class DatabaseUtility {
    public:
    // 
    explicit DatabaseUtility( Handle<DatabaseUtilitySession>& session );

    //
    DatabaseUtility( const DatabaseUtility& rhs );
    
    /// 
    virtual ~DatabaseUtility();

    /// 
    DatabaseUtility& operator=( const DatabaseUtility& rhs );
    
    ///
    std::set<std::string> listMappingVersions( const std::string& containerName );

    ///
    void importContainerSchema( const std::string& sourceConnectionString, const std::string& containerName );

    ///
    void importContainer( const std::string& sourceConnectionString, const std::string& containerName );

    ///
    void eraseMapping( const std::string& mappingVersion );
    
    private:

    Handle<DatabaseUtilitySession> m_session;
    
  };

}

#endif
