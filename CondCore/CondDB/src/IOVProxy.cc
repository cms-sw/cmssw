#include "CondCore/CondDB/interface/IOVProxy.h"
#include "SessionImpl.h"

namespace cond {

  namespace persistency {

    // implementation details...
    // only hosting data in this case
    class IOVProxyData {
    public: 
      
      IOVProxyData():
	iovSequence(){
      }
      
      // tag data
      std::string tag;
      cond::TimeType timeType;
      std::string payloadType;
      cond::SynchronizationType synchronizationType;
      cond::Time_t endOfValidity;
      cond::Time_t lastValidatedTime;
      // iov data
      cond::Time_t lowerGroup = cond::time::MAX_VAL;
      cond::Time_t higherGroup = cond::time::MIN_VAL;
      std::vector<cond::Time_t> sinceGroups;
      IOVProxy::IOVContainer iovSequence;
      size_t numberOfQueries = 0;
    };
    
    IOVProxy::Iterator::Iterator():
      m_current(),
      m_end(),
      m_timeType( cond::invalid ),
      m_endOfValidity(cond::time::MAX_VAL) {
    }
    
    IOVProxy::Iterator::Iterator( IOVContainer::const_iterator current, IOVContainer::const_iterator end, 
				  cond::TimeType timeType, cond::Time_t endOfValidity ):
      m_current( current ),
      m_end( end ),
      m_timeType( timeType ),
      m_endOfValidity( endOfValidity ){
    }
    
    IOVProxy::Iterator::Iterator( const Iterator& rhs ):
      m_current( rhs.m_current ),
      m_end( rhs.m_end ),
      m_timeType( rhs.m_timeType ),
      m_endOfValidity( rhs.m_endOfValidity ){
    }
    
    IOVProxy::Iterator& IOVProxy::Iterator::operator=( const Iterator& rhs ){
      if( this != &rhs ){
	m_current = rhs.m_current;
	m_end = rhs.m_end;
	m_timeType = rhs.m_timeType;
	m_endOfValidity = rhs.m_endOfValidity;
      }
      return *this;
    }
    
    cond::Iov_t IOVProxy::Iterator::operator*() {
      cond::Iov_t retVal;
      retVal.since = std::get<0>(*m_current);
      auto next = m_current;
      next++;
      
      // default is the end of validity when set...
      retVal.till = m_endOfValidity;
      // for the till, the next element has to be verified!
      if( next != m_end ){
	
	// the till has to be calculated according to the time type ( because of the packing for some types ) 
	retVal.till = cond::time::tillTimeFromNextSince( std::get<0>(*next), m_timeType );
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
    
    IOVProxy::IOVProxy( const std::shared_ptr<SessionImpl>& session ):
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
      
      checkTransaction( "IOVProxy::load" );
      std::string dummy;
      if(!m_session->iovSchema().tagTable().select( tag, m_data->timeType, m_data->payloadType, m_data->synchronizationType,
						    m_data->endOfValidity, dummy, m_data->lastValidatedTime ) ){
	throwException( "Tag \""+tag+"\" has not been found in the database.","IOVProxy::load");
      }
      m_data->tag = tag;
      
      // now get the iov sequence when required
      if( full ) {
	
	// load the full iov sequence in this case!
	m_session->iovSchema().iovTable().selectLatest( m_data->tag, m_data->iovSequence );
	m_data->lowerGroup = cond::time::MIN_VAL;
	m_data->higherGroup = cond::time::MAX_VAL;
      } else {
	m_session->iovSchema().iovTable().selectGroups( m_data->tag, m_data->sinceGroups );
      }
    }
    
    void IOVProxy::reload(){
      if(m_data.get() && !m_data->tag.empty()) load( m_data->tag );
    }
    
    void IOVProxy::reset(){
      if( m_data.get() ){
	m_data->lowerGroup = cond::time::MAX_VAL;
	m_data->higherGroup = cond::time::MIN_VAL;
	m_data->sinceGroups.clear();
	m_data->iovSequence.clear();
	m_data->numberOfQueries = 0;
      }
    }
    
    std::string IOVProxy::tag() const {
      return m_data.get() ? m_data->tag : std::string("");
    }
    
    cond::TimeType IOVProxy::timeType() const {
      return m_data.get() ? m_data->timeType : cond::invalid;
    }
    
    std::string IOVProxy::payloadObjectType() const {
      return m_data.get() ? m_data->payloadType : std::string("");
    }
    
    cond::SynchronizationType IOVProxy::synchronizationType() const {
      return m_data.get() ? m_data->synchronizationType : cond::SYNCHRONIZATION_UNKNOWN;
    }

    cond::Time_t IOVProxy::endOfValidity() const {
      return m_data.get() ? m_data->endOfValidity : cond::time::MIN_VAL;
    }
    
    cond::Time_t IOVProxy::lastValidatedTime() const {
      return m_data.get() ? m_data->lastValidatedTime : cond::time::MIN_VAL;
    }

    bool IOVProxy::isEmpty() const {
      return m_data.get() ? ( m_data->sinceGroups.size()==0 && m_data->iovSequence.size()==0 ) : true; 
    }
    
    void IOVProxy::checkTransaction( const std::string& ctx ) const {
      if( !m_session.get() ) throwException("The session is not active.",ctx );
      if( !m_session->isTransactionActive( false ) ) throwException("The transaction is not active.",ctx );
    }
    
    void IOVProxy::fetchSequence( cond::Time_t lowerGroup, cond::Time_t higherGroup ){
      m_data->iovSequence.clear();
      m_session->iovSchema().iovTable().selectLatestByGroup( m_data->tag, lowerGroup, higherGroup, m_data->iovSequence );
      
      m_data->lowerGroup = lowerGroup;
      m_data->higherGroup = higherGroup;
      
      m_data->numberOfQueries++;
    }
    
    IOVProxy::Iterator IOVProxy::begin() const {
      if( m_data.get() ){
	return Iterator( m_data->iovSequence.begin(), m_data->iovSequence.end(), 
			 m_data->timeType, m_data->endOfValidity );
      } 
      return Iterator();
    }
    
    IOVProxy::Iterator IOVProxy::end() const {
      if( m_data.get() ){
	return Iterator( m_data->iovSequence.end(), m_data->iovSequence.end(), m_data->timeType, m_data->endOfValidity );
      } 
      return Iterator();
    }
    
    // comparison functor for iov tuples: Time_t only and Time_t,string
    struct IOVComp {
      
      bool operator()( const cond::Time_t& x, const cond::Time_t& y ){ return (x < y); }
      
      bool operator()( const cond::Time_t& x, const std::tuple<cond::Time_t,cond::Hash>& y ){ return ( x < std::get<0>(y) ); }
    };
    
    // function to search in the vector the target time
    template <typename T> typename std::vector<T>::const_iterator search( const cond::Time_t& val, const std::vector<T>& container ){
      if( !container.size() ) return container.end();
      auto p = std::upper_bound( container.begin(), container.end(), val, IOVComp() );
      return (p!= container.begin()) ? p-1 : container.end();
    }
    
    IOVProxy::Iterator IOVProxy::find(cond::Time_t time) {
      checkTransaction( "IOVProxy::find" );
      // first check the available iov cache:
      // case 0 empty cache ( the first request )
      
      if( m_data->lowerGroup==cond::time::MAX_VAL ||
	  // case 1 : target outside
	  time < m_data->lowerGroup || time>= m_data->higherGroup ){
	
	// a new query required!
	// first determine the groups
	auto iGLow = search( time, m_data->sinceGroups );
	if( iGLow == m_data->sinceGroups.end() ){
	  // so suitable group=no iov at all! exiting...
	  return end();
	}
	auto iGHigh = iGLow;
	cond::Time_t lowG = 0;
	// unless the low group is the first one available, move the previous one to fully cover the interval
	if( iGLow != m_data->sinceGroups.begin() ){
	  iGLow--;
	  lowG = *iGLow;
	}
	// the upper group will be also extended to the next (covering in total up to three groups )
	iGHigh++;
	cond::Time_t highG = cond::time::MAX_VAL;
	if( iGHigh != m_data->sinceGroups.end() ) {
	  iGHigh++;
	  if( iGHigh != m_data->sinceGroups.end() ) highG = *iGHigh;
	}
	// finally, get the iovs for the selected group interval!!
	fetchSequence( lowG, highG );
      }
      
      // the current iov set is a good one...
      auto iIov = search( time, m_data->iovSequence );
      return Iterator( iIov, m_data->iovSequence.end(), m_data->timeType, m_data->endOfValidity );
    }
    
    cond::Iov_t IOVProxy::getInterval( cond::Time_t time ){
      Iterator valid = find( time );
      if( valid == end() ){
	throwException( "Can't find a valid interval for the specified time.","IOVProxy::getInterval");
      }
      return *valid;
    }

    cond::Iov_t IOVProxy::getLast(){
      checkTransaction( "IOVProxy::getLast" );
      cond::Iov_t ret;
      if( m_session->iovSchema().iovTable().getLastIov( m_data->tag, ret.since, ret.payloadId ) ){
	ret.till = cond::time::MAX_VAL;
      }
      return ret;
    }
    
    int IOVProxy::loadedSize() const {
      return m_data.get()? m_data->iovSequence.size() : 0;
    }
    
    int IOVProxy::sequenceSize() const {
      checkTransaction( "IOVProxy::sequenceSize" );
      size_t ret = 0;
      m_session->iovSchema().iovTable().getSize( m_data->tag, ret );
      return ret;
    }

    size_t IOVProxy::numberOfQueries() const {
      return m_data.get()?  m_data->numberOfQueries : 0;
    }
    
    std::pair<cond::Time_t,cond::Time_t> IOVProxy::loadedGroup() const {
      return m_data.get()? std::make_pair( m_data->lowerGroup, m_data->higherGroup ): std::make_pair( cond::time::MAX_VAL, cond::time::MIN_VAL );
    }

    const std::shared_ptr<SessionImpl>& IOVProxy::session() const {
      return m_session;
    }
    
  }
  
}
