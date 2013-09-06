#include "CondCore/CondDB/interface/IOVProxy.h"
#include "CondCore/CondDB/interface/TimeConversions.h"
#include "IOVSchema.h"
#include "SessionImpl.h"

namespace new_impl {

  // implementation details...
  // only hosting data in this case
  class IOVProxyData {
  public: 
   
    IOVProxyData():
      iovSequence(){
    }

    // tag data
    std::string tag;
    conddb::TimeType timeType;
    std::string payloadType;
    conddb::Time_t endOfValidity;
    conddb::Time_t lastValidatedTime;
    // iov data
    conddb::Time_t lowerGroup = conddb::time::MAX;
    conddb::Time_t higherGroup = conddb::time::MIN;
    std::vector<conddb::Time_t> sinceGroups;
    IOVProxy::IOVContainer iovSequence;
    size_t numberOfQueries = 0;
  };

IOVProxy::Iterator::Iterator():
  m_current(),
  m_end(),
  m_timeType( conddb::time::INVALID ){
}

IOVProxy::Iterator::Iterator( IOVContainer::const_iterator current, IOVContainer::const_iterator end, conddb::TimeType timeType ):
  m_current( current ),
  m_end( end ),
  m_timeType( timeType ){
}

IOVProxy::Iterator::Iterator( const Iterator& rhs ):
  m_current( rhs.m_current ),
  m_end( rhs.m_end ),
  m_timeType( rhs.m_timeType ){
}

IOVProxy::Iterator& IOVProxy::Iterator::operator=( const Iterator& rhs ){
  if( this != &rhs ){
    m_current = rhs.m_current;
    m_end = rhs.m_end;
    m_timeType = rhs.m_timeType;
  }
  return *this;
}

conddb::Iov_t IOVProxy::Iterator::operator*() {
  conddb::Iov_t retVal;
  retVal.since = std::get<0>(*m_current);
  auto next = m_current;
  next++;

  // for the till, the next element has to be verified!
  retVal.till = conddb::time::MAX;
  if( next != m_end ){

    // the till has to be calculated according to the time type ( because of the packing for some types ) 
    retVal.till = conddb::time::getTill( std::get<0>(*next), m_timeType );
  }
  retVal.payloadId = std::get<1>(*m_current);
  return retVal; 
}

IOVProxy::Iterator& IOVProxy::Iterator::operator++(){
  m_current++;
  return *this;
}

IOVProxy::Iterator IOVProxy::Iterator::operator++(int){
  Iterator tmp( *this );
  operator++();
  return tmp;
}

bool IOVProxy::Iterator::operator==( const Iterator& rhs ) const {
  if( m_current != rhs.m_current ) return false;
  if( m_end != rhs.m_end ) return false;
  return true;
}
      
bool IOVProxy::Iterator::operator!=( const Iterator& rhs ) const {
  return !operator==( rhs );
}

IOVProxy::IOVProxy():
  m_data(),
  m_session(){
}

IOVProxy::IOVProxy( const boost::shared_ptr<conddb::SessionImpl>& session ):
  m_data( new IOVProxyData ),
  m_session( session ){
}

IOVProxy::IOVProxy( const IOVProxy& rhs ):
  m_data( rhs.m_data ),
  m_session( rhs.m_session ){
}

IOVProxy& IOVProxy::operator=( const IOVProxy& rhs ){
  m_data = rhs.m_data;
  m_session = rhs.m_session;
  return *this;
}

void IOVProxy::load( const std::string& tag, bool full ){
  // clear
  reset();

  checkSession( "IOVProxy::load" );
  std::string dummy;
  if(!conddb::TAG::select( tag, m_data->timeType, m_data->payloadType, m_data->endOfValidity, dummy, m_data->lastValidatedTime, *m_session ) ){
    conddb::throwException( "Tag \""+tag+"\" has not been found in the database.","IOVProxy::load");
  }
  m_data->tag = tag;

  // now get the iov sequence when required
  if( full ) {

    // load the full iov sequence in this case!
    conddb::IOV::selectLast( m_data->tag, m_data->iovSequence, *m_session );
    m_data->lowerGroup = conddb::time::MIN;
    m_data->higherGroup = conddb::time::MAX;
  } else {
    conddb::IOV::selectGroups( m_data->tag, m_data->sinceGroups, *m_session );
  }
}

void IOVProxy::reload(){
  load( m_data->tag );
}

void IOVProxy::reset(){
  if( m_data.get() ){
    m_data->lowerGroup = conddb::time::MAX;
    m_data->higherGroup = conddb::time::MIN;
    m_data->sinceGroups.clear();
    m_data->iovSequence.clear();
    m_data->numberOfQueries = 0;
  }
}

std::string IOVProxy::tag() const {
  return m_data.get() ? m_data->tag : std::string("");
}

conddb::TimeType IOVProxy::timeType() const {
  return m_data.get() ? m_data->timeType : conddb::time::INVALID;
}

std::string IOVProxy::payloadObjectType() const {
  return m_data.get() ? m_data->payloadType : std::string("");
}

conddb::Time_t IOVProxy::endOfValidity() const {
  return m_data.get() ? m_data->endOfValidity : conddb::time::MIN;
}

conddb::Time_t IOVProxy::lastValidatedTime() const {
  return m_data.get() ? m_data->lastValidatedTime : conddb::time::MIN;
}

void IOVProxy::checkSession( const std::string& ctx ){
  if( !m_session.get() ) conddb::throwException("The session is not active.",ctx );
}

void IOVProxy::fetchSequence( conddb::Time_t lowerGroup, conddb::Time_t higherGroup ){
  m_data->iovSequence.clear();
  conddb::IOV::selectLastByGroup( m_data->tag, lowerGroup, higherGroup, m_data->iovSequence, *m_session );

  m_data->lowerGroup = lowerGroup;
  m_data->higherGroup = higherGroup;

  m_data->numberOfQueries++;
}

IOVProxy::Iterator IOVProxy::begin() const {
  if( m_data.get() ){
    return Iterator( m_data->iovSequence.begin(), m_data->iovSequence.end(), m_data->timeType );
  } 
  return Iterator();
}

IOVProxy::Iterator IOVProxy::end() const {
  if( m_data.get() ){
    return Iterator( m_data->iovSequence.end(), m_data->iovSequence.end(), m_data->timeType );
  } 
  return Iterator();
}

  // comparison functor for iov tuples: Time_t only and Time_t,string
  struct IOVComp {

    bool operator()( const conddb::Time_t& x, const conddb::Time_t& y ){ return (x < y); }

    bool operator()( const conddb::Time_t& x, const std::tuple<conddb::Time_t,conddb::Hash>& y ){ return ( x < std::get<0>(y) ); }
  };

  // function to search in the vector the target time
  template <typename T> typename std::vector<T>::const_iterator search( const conddb::Time_t& val, const std::vector<T>& container ){
    if( !container.size() ) return container.end();

    auto p = std::upper_bound( container.begin(), container.end(), val, IOVComp() );
    return (p!= container.begin()) ? p-1 : container.end();
  }

IOVProxy::Iterator IOVProxy::find(conddb::Time_t time) {
  checkSession( "IOVProxy::find" );
  // first check the available iov cache:
  // case 0 empty cache ( the first request )

  if( m_data->lowerGroup==conddb::time::MAX ||
      // case 1 : target outside
      time < m_data->lowerGroup || time>= m_data->higherGroup ){

    // a new query required!
    // first determine the groups
    auto iGLow = search( time, m_data->sinceGroups );
    if( iGLow == m_data->sinceGroups.end() ){
      return end();
    }
    auto iGHigh = iGLow;
    conddb::Time_t lowG = 0;
    if( iGLow != m_data->sinceGroups.begin() ){
      iGLow--;
      lowG = *iGLow;
    }
    iGHigh++;
    conddb::Time_t highG = conddb::time::MAX;
    if( iGHigh != m_data->sinceGroups.end() ) {
      iGHigh++;
      if( iGHigh != m_data->sinceGroups.end() ) highG = *iGHigh;
    }
    fetchSequence( lowG, highG );
  }
  
  /**
  bool load = false;
  if( !m_data->iovSequence.empty() ){
    conddb::Time_t front=std::get<0>(m_data->iovSequence.front() );
    if( time < front || time > front+2*conddb::time::SINCE_GROUP_SIZE ) load = true;
  } else {
    load = true;
  }
  if( load ){
    // a (new) query required...
    // fetch the iov sequence subset for the two groups
    fetchSequence( time );    
  }
  **/

  // the current iov set is a good one...
  auto iIov = search( time, m_data->iovSequence );
  return Iterator( iIov, m_data->iovSequence.end(), m_data->timeType );
}

conddb::Iov_t IOVProxy::getInterval( conddb::Time_t time ){
  Iterator valid = find( time );
  if( valid == end() ){
    conddb::throwException( "Can't find a valid interval for the specified time.","IOVProxy::getInterval");
  }
  return *valid;
}


int IOVProxy::size() const {
  return m_data.get()? m_data->iovSequence.size() : 0;
}

size_t IOVProxy::numberOfQueries() const {
  return m_data.get()?  m_data->numberOfQueries : 0;
}

std::pair<conddb::Time_t,conddb::Time_t> IOVProxy::loadedGroup() const {
  return m_data.get()? std::make_pair( m_data->lowerGroup, m_data->higherGroup ): std::make_pair( conddb::time::MAX, conddb::time::MIN );
}


}
