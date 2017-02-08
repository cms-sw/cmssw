#ifndef INCLUDE_ORA_TRANSACTIONCACHE_H
#define INCLUDE_ORA_TRANSACTIONCACHE_H

#include "CondCore/ORA/interface/Handle.h"
//
#include <string>
#include <map>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

namespace ora {

  class DatabaseContainer;
  class DatabaseUtilitySession;

  class TransactionCache {
    public:
    TransactionCache();

    virtual ~TransactionCache();

    void clear();

    void setDbExists( bool exists );

    bool dbExistsLoaded();

    bool dbExists();

    void addContainer( int id, const std::string& name, Handle<DatabaseContainer>& contPtr );

    void eraseContainer( int id, const std::string& name );

    Handle<DatabaseContainer> getContainer( int id );
    
    Handle<DatabaseContainer> getContainer( const std::string& name );

    const std::map<int, Handle<DatabaseContainer> >& containers();

    void dropDatabase();

    std::map<std::string,std::string>& dbParams();

    void setUtility( Handle<DatabaseUtilitySession>& utility );

    Handle<DatabaseUtilitySession> utility();

    bool isLoaded();

    void setLoaded();

    void setNamedReference( const std::string& name, boost::shared_ptr<void>& data );

    boost::shared_ptr<void> getNamedReference( const std::string& name );

    void setDropPermission( bool allowed );
    bool dropPermissionLoaded();
    bool dropPermission();

    private:
    void cleanUpNamedRefCache();

    private:
    std::pair<bool,bool> m_dbExists;
    std::map<std::string, int> m_containersByName;
    std::map<int, Handle<DatabaseContainer> > m_containersById;
    std::map<std::string,std::string> m_dbParams;
    Handle<DatabaseUtilitySession> m_utility;
    bool m_loaded;
    std::map<std::string,boost::weak_ptr<void> > m_namedRefCache;
    std::pair<bool,bool> m_dropPermission;
  };
}

#endif
