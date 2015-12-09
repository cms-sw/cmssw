#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/IOVProxy.h"
#include "CondCore/CondDB/interface/GTProxy.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include <iostream>

#include <chrono>

// ================================================================================

class Timer {

public:
    Timer( const std::string &nameIn ) : name(nameIn) { reset(); }
    void reset() { start = std::chrono::steady_clock::now(); intervals.clear(); intervalNames.clear(); interval("start"); }

    void interval(const std::string & intName ) { intervals.push_back(std::chrono::steady_clock::now()); intervalNames.push_back(intName); }

    void fetchInt(size_t sizeIn) { fetchTime.push_back(std::chrono::steady_clock::now()); fetchNum.push_back(sizeIn); }
    void deserInt(size_t sizeIn) { deserTime.push_back(std::chrono::steady_clock::now()); deserNum.push_back(sizeIn); }

    void show(std::ostream &os=std::cout) { showIntervals(os); showFetchInfo(os); showDeserInfo(os); }
    void showIntervals(std::ostream &os=std::cout);
    void showFetchInfo (std::ostream &os=std::cout);
    void showDeserInfo (std::ostream &os=std::cout);

private:
    
    std::string name;
    
    std::chrono::time_point<std::chrono::steady_clock> start;

    std::vector< std::chrono::time_point<std::chrono::steady_clock> > intervals;
    std::vector< std::string > intervalNames;

    std::vector< std::chrono::time_point<std::chrono::steady_clock> > fetchTime;
    std::vector< int > fetchNum;

    std::vector< std::chrono::time_point<std::chrono::steady_clock> > deserTime;
    std::vector< int > deserNum;

};

void Timer::showIntervals(std::ostream &os) {
    
    os << std::endl;
    os << "Serialization type: " << name << std::endl;
    for (size_t i=1; i<intervals.size(); i++) {
        os << intervalNames[i] << " : " << std::chrono::duration<double, std::milli>(intervals[i] - intervals[i-1]).count() << " msec. " << std::endl;        
    }
    os << "\noverall time elapsed" << " : " << std::chrono::duration<double, std::milli>(intervals[intervals.size()-1] - intervals[0]).count() << " msec. " << std::endl;
    os << std::endl;    
}

void Timer::showFetchInfo(std::ostream &os) {
    os << std::endl;
    os << "Serialization type: " << name << std::endl;
    if (fetchTime.size() < 1) {
      os << "No fetch info available." << std::endl;
      return;
    }
    int totSize = 0;
    for (size_t i=1; i<fetchTime.size(); i++) {
        totSize += fetchNum[i];
        auto delta = std::chrono::duration<double, std::milli>(fetchTime[i] - fetchTime[i-1]).count();
        os << fetchNum[i] << " : " << delta << " ms (" << float(fetchNum[i])/(1024.*float(delta)) << " MB/s)" << std::endl;
    }
    auto deltaAll = std::chrono::duration<double, std::milli>(fetchTime[fetchTime.size()-1] - fetchTime[0]).count();
    os << "\noverall time for "<< totSize << " bytes : " << deltaAll << " ms (" << float(totSize)/(1024.*float(deltaAll)) << " MB/s)" << std::endl;
    os << std::endl;    
}

void Timer::showDeserInfo(std::ostream &os) {
    os << std::endl;
    os << "Serialization type: " << name << std::endl;
    if (deserTime.size() < 1) {
      os << "No deserialization info available." << std::endl;
      return;
    }
    int totSize = 0;
    for (size_t i=1; i<deserTime.size(); i++) {
        totSize += deserNum[i];
        auto delta = std::chrono::duration<double, std::milli>(deserTime[i] - deserTime[i-1]).count();
        os << deserNum[i] << " : " << delta << " ms (" << float(deserNum[i])/(1024.*float(delta)) << " MB/s)" << std::endl;
    }
    auto deltaAll = std::chrono::duration<double, std::milli>(deserTime[deserTime.size()-1] - deserTime[0]).count();
    os << "\noverall time for "<< totSize << " bytes : " << deltaAll << " ms (" << float(totSize)/(1024.*float(deltaAll)) << " MB/s)" << std::endl;
    os << std::endl;    
}

// ================================================================================


namespace cond {

  using namespace persistency;

  class UntypedPayloadProxy {
  public:
    explicit UntypedPayloadProxy( Session& session );

    UntypedPayloadProxy( const UntypedPayloadProxy& rhs );

    UntypedPayloadProxy& operator=( const UntypedPayloadProxy& rhs );

    void load( const std::string& tag );

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
  m_session( session ){
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
  m_session.transaction().start();
  m_iov = m_session.readIov( tag );
  m_session.transaction().commit();
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
    m_session.transaction().start();

    auto iIov = m_iov.find( targetTime );
    if(iIov == m_iov.end() ) cond::throwException(std::string("Tag ")+m_iov.tag()+": No iov available for the target time:"+boost::lexical_cast<std::string>(targetTime),"UntypedPayloadProxy::get");
    m_data->current = *iIov;
    event <<"For target time "<<targetTime<<" got a valid since:"<<m_data->current.since<<" from group ["<<m_iov.loadedGroup().first<<" - "<<m_iov.loadedGroup().second<<"]"; 

    std::string payloadType("");
    Binary data; 
    Binary streamerInfo; 
    loaded = m_session.fetchPayloadData( m_data->current.payloadId, payloadType, data, streamerInfo );
    m_session.transaction().commit();
    if( !loaded ){
      std::cout <<"ERROR: payload with id "<<m_data->current.payloadId<<" could not be loaded."<<std::endl;
    }else {
      std::stringstream sz;
      sz << data.size();
      if( debug ) std::cout <<"Loaded payload of type \""<< payloadType <<"\" ("<<sz.str()<<" bytes)"<<std::endl;  
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
  size_t n = 1;
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

  Timer timex("condDBv1");

  ConnectionPool connPool;
  if( hasDebug() ) connPool.setMessageVerbosity( coral::Debug );
  connPool.configure();
  Session session = connPool.createSession( connect );
  session.transaction().start();
  
  std::cout <<"Loading Global Tag "<<gtag<<std::endl;
  GTProxy gt = session.readGlobalTag( gtag );

  session.transaction().commit();

  std::cout <<"Loading "<<gt.size()<<" tags..."<<std::endl;
  std::vector<UntypedPayloadProxy> proxies;
  std::map<std::string,size_t> requests;
  for( auto t: gt ){
    std::pair<std::string,std::string> tagParams = parseTag( t.tagName() );
    std::string tagConnStr = connect;
    Session tagSession = session; 
    if( !tagParams.second.empty() ) {
      tagConnStr = tagParams.second;
      tagSession = connPool.createSession( tagConnStr );
    }
    UntypedPayloadProxy p( tagSession );
    try{
      p.load( tagParams.first );
      proxies.push_back( p );
      requests.insert( std::make_pair( tagParams.first, 0 ) );
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
      if( r != requests.end() ){
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
  }

  timex.interval("iterations done");
  timex.showIntervals();

  std::cout <<std::endl;
  std::cout <<"*** End of job."<<std::endl;
  std::cout <<"*** GT: "<<gtag<<" Tags:"<<gt.size()<<" Loaded:"<<proxies.size()<<std::endl;
  std::cout<<std::endl;
  if( verbose ){
    for( auto p: proxies ){
      auto r = requests.find( p.tag() );
      if( r != requests.end() ){
        std::cout <<"*** Tag: "<<p.tag()<<" Requests processed:"<<r->second<<" Queries:"<<p.numberOfQueries()<<std::endl;
        if( verbose ){
    	const std::vector<std::string>& hist = p.history();
    	for( auto e: p.history() ) std::cout <<"    "<<e<<std::endl;
        }
      }
    }
  }

  return 0;
}

int main( int argc, char** argv ){

  cond::TestGTLoad test;
  return test.run(argc,argv);
}

