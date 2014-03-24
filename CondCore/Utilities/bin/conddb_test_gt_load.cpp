#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/IOVProxy.h"
#include "CondCore/CondDB/interface/GTProxy.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include <iostream>

namespace cond {

  using namespace persistency;

  class UntypedPayloadProxy {
  public:
    explicit UntypedPayloadProxy( Session& session );

    UntypedPayloadProxy( const UntypedPayloadProxy& rhs );

    UntypedPayloadProxy& operator=( const UntypedPayloadProxy& rhs );

    void load( const std::string& tag );

    void reload();

    void reset();

    TimeType timeType() const;
    std::string tag() const;

    bool get( cond::Time_t targetTime, bool debug );

    size_t numberOfQueries() const;

    const std::vector<std::string>& history() const;

  private:
  
    struct pimpl {
      cond::Iov_t current;
      std::vector<std::string> history;
    };
    
    Session m_session;
    IOVProxy m_iov;
    boost::shared_ptr<pimpl> m_data;
  };

  class TestGTLoad : public cond::Utilities {
    public:
      TestGTLoad();
      int execute();
  };
}

cond::UntypedPayloadProxy::UntypedPayloadProxy( Session& session ):
  m_session( session ),
  m_iov( session.iovProxy() ),
  m_data(){
  m_data.reset( new pimpl );
  m_data->current.clear();
}

cond::UntypedPayloadProxy::UntypedPayloadProxy( const UntypedPayloadProxy& rhs ):
  m_session( rhs.m_session ),
  m_iov( rhs.m_iov ),
  m_data( rhs.m_data ){
}

cond::UntypedPayloadProxy& cond::UntypedPayloadProxy::operator=( const cond::UntypedPayloadProxy& rhs ){
  m_session = rhs.m_session;
  m_iov = rhs.m_iov;
  m_data = rhs.m_data;
  return *this;
}

void cond::UntypedPayloadProxy::load( const std::string& tag ){
  m_data->current.clear();
  m_iov.load( tag );
}

void cond::UntypedPayloadProxy::reload(){
  m_data->current.clear();
  m_iov.reload();
}

void cond::UntypedPayloadProxy::reset(){
  m_iov.reset();
  m_data->current.clear();
}

std::string cond::UntypedPayloadProxy::tag() const {
  return m_iov.tag();
}

cond::TimeType cond::UntypedPayloadProxy::timeType() const {
  return m_iov.timeType();
}


bool cond::UntypedPayloadProxy::get( cond::Time_t targetTime, bool debug ){
  bool loaded = false;
  std::stringstream event;

  //  check if the current iov loaded is the good one...
  if( targetTime < m_data->current.since || targetTime >= m_data->current.till ){

    // a new payload is required!
    if( debug )std::cout <<" Searching tag "<<m_iov.tag()<<" for a valid payload for time="<<targetTime<<std::endl;
    auto iIov = m_iov.find( targetTime );
    if(iIov == m_iov.end() ) cond::throwException(std::string("Tag ")+m_iov.tag()+": No iov available for the target time:"+boost::lexical_cast<std::string>(targetTime),"UntypedPayloadProxy::get");
    m_data->current = *iIov;
    event <<"For target time "<<targetTime<<" got a valid since:"<<m_data->current.since<<" from group ["<<m_iov.loadedGroup().first<<" - "<<m_iov.loadedGroup().second<<"]"; 

    std::string payloadType("");
    Binary data; 
    loaded = m_session.fetchPayloadData( m_data->current.payloadId, payloadType, data );
    if( !loaded ){
      std::cout <<"ERROR: payload with id "<<m_data->current.payloadId<<" could not be loaded."<<std::endl;
    }else {
      if( debug ) std::cout <<"Loaded payload of type \""<< payloadType <<"\" ("<<data.size()<<" bytes)"<<std::endl;  
    }
  } else {
    event <<"Target time "<<targetTime<<" is within range for payloads available in cache: ["<<m_data->current.since<<" - "<<m_data->current.till<<"]";
  }
  m_data->history.push_back( event.str() );
  return loaded;
}

size_t cond::UntypedPayloadProxy::numberOfQueries() const {
  return m_iov.numberOfQueries();
}

const std::vector<std::string>& cond::UntypedPayloadProxy::history() const {
  return m_data->history;
}

cond::TestGTLoad::TestGTLoad():
  Utilities("conddb_test_gt_load"){
  addConnectOption("connect","c","database connection string(required)");
  addAuthenticationOptions();
  addOption<size_t>("iterations","n","number of iterations (default=10)");
  addOption<Time_t>("start_run","R","start for Run iterations (default=150005)");
  addOption<Time_t>("step_run","r","step for Run iterations (default=1000)");
  addOption<Time_t>("start_ts","T","start for TS iterations (default=5800013687234232320)");
  addOption<Time_t>("step_ts","t","step for TS iterations (default=10000000000000)");
  addOption<Time_t>("start_lumi","L","start for Lumi iterations (default=908900979179966)");
  addOption<Time_t>("step_lumi","l","step for Lumi iterations (default=10000000000)");
  addOption<std::string>("globaltag","g","global tag (required)");
  addOption<bool>("verbose","v","verbose print out (optional)");
}

int cond::TestGTLoad::execute(){

  std::string gtag = getOptionValue<std::string>("globaltag");
  bool debug = hasDebug();
  std::string connect = getOptionValue<std::string>("connect");
  bool verbose = hasOptionValue("verbose");
  size_t n = 10;
  if(hasOptionValue("iterations")) n = getOptionValue<size_t>("iterations"); 
  Time_t startRun= 150005;
  if(hasOptionValue("start_run")) startRun = getOptionValue<Time_t>("start_run");
  Time_t stepRun= 1000;
  if(hasOptionValue("step_run")) stepRun = getOptionValue<Time_t>("step_run");
  Time_t startTs= 5800013687234232320;
  if(hasOptionValue("start_ts")) startTs = getOptionValue<Time_t>("start_ts");
  Time_t stepTs= 10000000000000;
  if(hasOptionValue("step_ts")) stepTs = getOptionValue<Time_t>("step_ts");
  Time_t startLumi= 908900979179966;
  if(hasOptionValue("start_lumi")) startLumi = getOptionValue<Time_t>("start_lumi");
  Time_t stepLumi= 10000000000;
  if(hasOptionValue("step_lumi")) stepLumi = getOptionValue<Time_t>("step_lumi");

  initializePluginManager();

  ConnectionPool connPool;
  if( hasDebug() ) connPool.setMessageVerbosity( coral::Debug );
  Session session = connPool.createSession( connect );
  session.transaction().start();
  
  std::cout <<"Loading Global Tag "<<gtag<<std::endl;
  GTProxy gt = session.readGlobalTag( gtag );

  std::cout <<"Loading "<<gt.size()<<" tags..."<<std::endl;
  std::vector<UntypedPayloadProxy> proxies;
  std::map<std::string,size_t> requests;
  for( auto t: gt ){
    UntypedPayloadProxy p( session );
    try{
      p.load( t.tagName() );
      proxies.push_back( p );
      requests.insert( std::make_pair( t.tagName(), 0 ) );
    } catch ( const cond::Exception& e ){
      std::cout <<"ERROR: "<<e.what()<<std::endl;
    }
  }
  std::cout<<proxies.size()<<" tags successfully loaded."<<std::endl;
  
  std::cout <<"Iterating on "<<n<<" IOV request(s)..."<<std::endl;

  for( size_t i=0;i<n; i++ ){
    Time_t run = startRun+i*stepRun;
    Time_t lumi = startLumi +i*stepLumi;
    Time_t ts = startTs+i*stepTs;
    for( auto p: proxies ){
      bool loaded = false;
      time::TimeType ttype = p.timeType();
      auto r = requests.find( p.tag() );
      try{
	if( ttype==runnumber ){
	  p.get( run, hasDebug() );	
	  r->second++;
	} else if( ttype==lumiid ){
	  p.get( lumi, hasDebug() );
	  r->second++;
	} else if( ttype==timestamp){
	  p.get( ts, hasDebug() );
	  r->second++;
	} else {
	  std::cout <<"WARNING: iov request on tag "<<p.tag()<<" (timeType="<<time::timeTypeName(p.timeType())<<") has been skipped."<<std::endl;
	}
      } catch ( const cond::Exception& e ){
	std::cout <<"ERROR:"<<e.what()<<std::endl;
      }
    }
  }

  session.transaction().commit();

  std::cout <<std::endl;
  std::cout <<"*** End of job."<<std::endl;
  std::cout <<"*** GT: "<<gtag<<" Tags:"<<gt.size()<<" Loaded:"<<proxies.size()<<std::endl;
  std::cout<<std::endl;
  for( auto p: proxies ){
    auto r = requests.find( p.tag() );
    std::cout <<"*** Tag: "<<p.tag()<<" Requests processed:"<<r->second<<" Queries:"<<p.numberOfQueries()<<std::endl;
    if( verbose ){
      const std::vector<std::string>& hist = p.history();
      for( auto e: p.history() ) std::cout <<"    "<<e<<std::endl;
    }
  }

  return 0;
}

int main( int argc, char** argv ){

  cond::TestGTLoad test;
  return test.run(argc,argv);
}

