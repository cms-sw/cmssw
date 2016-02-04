#include "TransactionCache.h"
#include "DatabaseContainer.h"
#include "DatabaseUtilitySession.h"

ora::TransactionCache::TransactionCache():
  m_dbExists(false,false),
  m_containersByName(),
  m_containersById(),
  m_dbParams(),
  m_utility(),
  m_loaded( false ),
  m_namedRefCache(),
  m_dropPermission(false,false){
}

ora::TransactionCache::~TransactionCache(){
  clear();
}

void ora::TransactionCache::clear(){
  m_dbExists.first = false;
  m_dbExists.second = false;
  for( std::map<int, Handle<DatabaseContainer> >::iterator iCont = m_containersById.begin();
       iCont != m_containersById.end(); iCont++ ){
    iCont->second.clear();
  }
  m_utility.clear();
  m_containersById.clear();
  m_containersByName.clear();
  m_dbParams.clear();
  m_loaded = false;
  m_namedRefCache.clear();
  m_dropPermission.first = false;
  m_dropPermission.second = false;
}

void ora::TransactionCache::setDbExists( bool exists ){
  m_dbExists.first = true;
  m_dbExists.second = exists;  
}

bool ora::TransactionCache::dbExistsLoaded(){
  return m_dbExists.first;  
}

bool ora::TransactionCache::dbExists(){
  return m_dbExists.second;
}

void ora::TransactionCache::addContainer( int id,
                                          const std::string& name,
                                          Handle<DatabaseContainer>& contPtr ){
  m_containersById.insert( std::make_pair( id, contPtr ) );
  m_containersByName.insert( std::make_pair( name, id ) );
}

void ora::TransactionCache::eraseContainer( int id,
                                            const std::string& name ){
  m_containersById.erase( id );
  m_containersByName.erase( name );
}
      
ora::Handle<ora::DatabaseContainer> ora::TransactionCache::getContainer( int id ){
  Handle<DatabaseContainer> instance;
  std::map<int, Handle<DatabaseContainer> >::iterator iCont = m_containersById.find( id );
  if( iCont != m_containersById.end() ){
    instance = iCont->second;
  }
  return instance;
}

ora::Handle<ora::DatabaseContainer> ora::TransactionCache::getContainer( const std::string& name ){
  Handle<DatabaseContainer> instance;
  std::map<std::string, int>::iterator iId = m_containersByName.find( name );
  if( iId == m_containersByName.end() ){
    return instance;
  }
  return getContainer( iId->second );
}

const std::map<int,ora::Handle<ora::DatabaseContainer> >& ora::TransactionCache::containers(){
  return m_containersById;  
}

void ora::TransactionCache::dropDatabase(){
  m_dbExists.second = true;
  m_dbExists.second = false;
  m_containersById.clear();
  m_containersByName.clear();
  m_dbParams.clear();
  m_namedRefCache.clear();  
  m_dropPermission.first = false;
  m_dropPermission.second = false;
}

std::map<std::string,std::string>& ora::TransactionCache::dbParams(){
  return m_dbParams;
}

void ora::TransactionCache::setUtility( Handle<DatabaseUtilitySession>& utility ){
  m_utility = utility;
}

ora::Handle<ora::DatabaseUtilitySession> ora::TransactionCache::utility(){
  return m_utility;
}

bool ora::TransactionCache::isLoaded(){
  return m_loaded;
}

void ora::TransactionCache::setLoaded(){
  m_loaded = true;
}

void ora::TransactionCache::cleanUpNamedRefCache(){
  std::vector<std::string> namesToDelete;
  for( std::map<std::string,boost::weak_ptr<void> >::const_iterator iEntry = m_namedRefCache.begin();
       iEntry != m_namedRefCache.end(); iEntry++ ){
    if( iEntry->second.expired() ) namesToDelete.push_back( iEntry->first );
  }
  for( std::vector<std::string>::const_iterator iName = namesToDelete.begin();
       iName != namesToDelete.end(); iName++ ){
    m_namedRefCache.erase( *iName );
  }
}

void ora::TransactionCache::setNamedReference( const std::string& name, 
                                               boost::shared_ptr<void>& data ){
  m_namedRefCache.insert( std::make_pair( name, boost::weak_ptr<void>(data) ) );  
}

boost::shared_ptr<void> ora::TransactionCache::getNamedReference( const std::string& name ){
  cleanUpNamedRefCache();
  boost::shared_ptr<void> ret;
  std::map<std::string,boost::weak_ptr<void> >::const_iterator iEntry = m_namedRefCache.find( name );
  if( iEntry != m_namedRefCache.end() ){
    ret = iEntry->second.lock();
  }
  return ret;
}

void ora::TransactionCache::setDropPermission( bool allowed ){
  m_dropPermission.first = true;
  m_dropPermission.second = allowed;
}

bool ora::TransactionCache::dropPermissionLoaded(){
  return m_dropPermission.first;
}

bool ora::TransactionCache::dropPermission(){
  return m_dropPermission.second;
}
