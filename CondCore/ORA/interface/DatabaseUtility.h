#ifndef INCLUDE_ORA_DATABASEUTILITY_H
#define INCLUDE_ORA_DATABASEUTILITY_H

#include "Handle.h"
#include <boost/shared_ptr.hpp>
//
#include <set>
#include <map>

namespace ora {

  class DatabaseUtilitySession;
  
  class DatabaseUtility {
    public:
    // 
    DatabaseUtility();

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
    std::map<std::string,std::string> listMappings( const std::string& containerName );

    ///
    bool dumpMapping( const std::string& mappingVersion, std::ostream& outputStream );
    
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
