#include "CondCore/CondDB/interface/GTProxy.h"
#include "GTSchema.h"
#include "SessionImpl.h"

namespace new_impl {

  // implementation details...
  // only hosting data in this case
  class GTProxyData {
  public: 
   
    GTProxyData():
      tagList(){
    }

    std::string name;
    conddb::Time_t validity;
    boost::posix_time::ptime snapshotTime;
    // tag list
    GTProxy::GTContainer tagList;
  };

GTProxy::Iterator::Iterator():
  m_current(){
}

GTProxy::Iterator::Iterator( GTContainer::const_iterator current ):
  m_current( current ){
}

GTProxy::Iterator::Iterator( const Iterator& rhs ):
  m_current( rhs.m_current ){
}

GTProxy::Iterator& GTProxy::Iterator::operator=( const Iterator& rhs ){
  if( this != &rhs ){
    m_current = rhs.m_current;
  }
  return *this;
}

conddb::GTEntry_t GTProxy::Iterator::operator*() {
  return conddb::GTEntry_t( *m_current );
}

GTProxy::Iterator& GTProxy::Iterator::operator++(){
  m_current++;
  return *this;
}

GTProxy::Iterator GTProxy::Iterator::operator++(int){
  Iterator tmp( *this );
  operator++();
  return tmp;
}

bool GTProxy::Iterator::operator==( const Iterator& rhs ) const {
  if( m_current != rhs.m_current ) return false;
  return true;
}
      
bool GTProxy::Iterator::operator!=( const Iterator& rhs ) const {
  return !operator==( rhs );
}

GTProxy::GTProxy():
  m_data(),
  m_session(){
}

GTProxy::GTProxy( const std::shared_ptr<conddb::SessionImpl>& session ):
  m_data( new GTProxyData ),
  m_session( session ){
}

GTProxy::GTProxy( const GTProxy& rhs ):
  m_data( rhs.m_data ),
  m_session( rhs.m_session ){
}

GTProxy& GTProxy::operator=( const GTProxy& rhs ){
  m_data = rhs.m_data;
  m_session = rhs.m_session;
  return *this;
}

void GTProxy::load( const std::string& gtName ){
  // clear
  reset();

  checkSession( "GTProxy::load" );

  if(!conddb::GLOBAL_TAG::select( gtName, m_data->validity, m_data->snapshotTime, *m_session ) ){
    conddb::throwException( "Global Tag \""+gtName+"\" has not been found in the database.","GTProxy::load");
  }
  m_data->name = gtName;

    // load the full iov sequence in this case!
  conddb::GLOBAL_TAG_MAP::select( m_data->name, m_data->tagList, *m_session );

}

void GTProxy::reload(){
  load( m_data->name );
}

void GTProxy::reset(){
  if( m_data.get() ){
    m_data->tagList.clear();
  }
}

std::string GTProxy::name() const {
  return m_data.get() ? m_data->name : std::string("");
}

conddb::Time_t GTProxy::validity() const {
  return m_data.get() ? m_data->validity : conddb::time::MIN;
}

boost::posix_time::ptime GTProxy::snapshotTime() const {
  return m_data.get() ? m_data->snapshotTime : boost::posix_time::ptime();
}

void GTProxy::checkSession( const std::string& ctx ){
  if( !m_session.get() ) conddb::throwException("The session is not active.",ctx );
}

GTProxy::Iterator GTProxy::begin() const {
  if( m_data.get() ){
    return Iterator( m_data->tagList.begin() );
  } 
  return Iterator();
}

GTProxy::Iterator GTProxy::end() const {
  if( m_data.get() ){
    return Iterator( m_data->tagList.end() );
  } 
  return Iterator();
}

int GTProxy::size() const {
  return m_data.get()? m_data->tagList.size() : 0;
}

}
