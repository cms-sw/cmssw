#include "CondCore/CondDB/interface/RunInfoProxy.h"
#include "SessionImpl.h"

namespace cond {

  namespace persistency {

    // implementation details...
    // only hosting data in this case
    class RunInfoProxyData {
    public: 
      
      RunInfoProxyData():
	runList(){
      }
      
      // the data loaded
      RunInfoProxy::RunInfoData runList;
    };

    RunInfoProxy::Iterator::Iterator():
      m_current(){
    }

    RunInfoProxy::Iterator::Iterator( RunInfoData::const_iterator current ):
      m_current( current ){
    }
    
    RunInfoProxy::Iterator::Iterator( const Iterator& rhs ):
      m_current( rhs.m_current ){
    }
    
    RunInfoProxy::Iterator& RunInfoProxy::Iterator::operator=( const Iterator& rhs ){
      if( this != &rhs ){
	m_current = rhs.m_current;
      }
      return *this;
    }

    cond::RunInfo_t RunInfoProxy::Iterator::operator*() {
      return cond::RunInfo_t( *m_current );
    }
    
    RunInfoProxy::Iterator& RunInfoProxy::Iterator::operator++(){
      m_current++;
      return *this;
    }
    
    RunInfoProxy::Iterator RunInfoProxy::Iterator::operator++(int){
      Iterator tmp( *this );
      operator++();
      return tmp;
    }
    
    bool RunInfoProxy::Iterator::operator==( const Iterator& rhs ) const {
      if( m_current != rhs.m_current ) return false;
      return true;
    }
    
    bool RunInfoProxy::Iterator::operator!=( const Iterator& rhs ) const {
      return !operator==( rhs );
    }
    
    RunInfoProxy::RunInfoProxy():
      m_data(),
      m_session(){
    }
    
    RunInfoProxy::RunInfoProxy( const std::shared_ptr<SessionImpl>& session ):
      m_data( new RunInfoProxyData ),
      m_session( session ){
    }
    
    RunInfoProxy::RunInfoProxy( const RunInfoProxy& rhs ):
      m_data( rhs.m_data ),
      m_session( rhs.m_session ){
    }
    
    RunInfoProxy& RunInfoProxy::operator=( const RunInfoProxy& rhs ){
      m_data = rhs.m_data;
      m_session = rhs.m_session;
      return *this;
    }
    
    //
    bool RunInfoProxy::load( Time_t low, Time_t up ){
      // clear
      reset();
      
      checkTransaction( "RunInfoProxy::load" );
      
      return runinfo::getRunStartTime( m_session->coralSession->schema( runinfo::RUNINFO_SCHEMA ), low, up, m_data->runList );
    }

    void RunInfoProxy::reset(){
      if( m_data.get() ){
	m_data->runList.clear();
      }
    }
    
    void RunInfoProxy::checkTransaction( const std::string& ctx ){
      if( !m_session.get() ) throwException("The session is not active.",ctx );
      if( !m_session->isTransactionActive( false ) ) throwException("The transaction is not active.",ctx );
    }
    
    RunInfoProxy::Iterator RunInfoProxy::begin() const {
      if( m_data.get() ){
	return Iterator( m_data->runList.begin() );
      } 
      return Iterator();
    }
    
    RunInfoProxy::Iterator RunInfoProxy::end() const {
      if( m_data.get() ){
	return Iterator( m_data->runList.end() );
      } 
      return Iterator();
    }

    // comparison functor for iov tuples: Time_t only and Time_t,string
    struct IOVComp {
      
      bool operator()( const std::tuple<cond::Time_t,boost::posix_time::ptime>& x, const cond::Time_t& y ){ return ( y > std::get<0>(x) ); }
      
    };
    
    RunInfoProxy::Iterator RunInfoProxy::find( Time_t target ) const {
      if( m_data.get() ){
	auto p = std::lower_bound( m_data->runList.begin(), m_data->runList.end(), target, IOVComp() );
        return Iterator( p );
      } 
      return Iterator();
    }
    
    int RunInfoProxy::size() const {
      return m_data.get()? m_data->runList.size() : 0;
    }

  }
}
