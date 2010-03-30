#include "TransactionCache.h"
#include "DatabaseContainer.h"

ora::TransactionCache::TransactionCache():
  m_dbExists(false,false),
  m_containersByName(),
  m_containersById(),
  m_loaded( false ){
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
  m_containersById.clear();
  m_containersByName.clear();
  m_loaded = false;
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

bool ora::TransactionCache::isLoaded(){
  return m_loaded;
}

void ora::TransactionCache::setLoaded(){
  m_loaded = true;
}

