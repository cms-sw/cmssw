#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/IOVProxy.h"
#include "CondCore/CondDB/interface/GTProxy.h"

#include "CondCore/CondDB/src/IOVSchema.cc"

#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/Utilities/interface/CondDBImport.h"
#include <iostream>
#include <fstream>

#include <chrono>
#include <memory>

#include <boost/thread/mutex.hpp>
#include "tbb/parallel_for_each.h"
#include "tbb/task_scheduler_init.h"

namespace cond {

  using namespace persistency;

  class ConnectionPoolWrapper {
  public:
    ConnectionPoolWrapper( int authenticationSystem, const std::string& authenticationPath, bool debug );
    Session createSession( const std::string& connectionString );
    boost::mutex lock;
    ConnectionPool connPool;
  };

  class UntypedPayloadProxy {
  public:
    UntypedPayloadProxy();

    UntypedPayloadProxy( const UntypedPayloadProxy& rhs );

    UntypedPayloadProxy& operator=( const UntypedPayloadProxy& rhs );

    void init( Session session );

    void load( const std::string& tag );

    void reload();

    void reset();

    void disconnect();

    TimeType timeType() const;
    std::string tag() const;
    std::string payloadType() const;

    bool get( cond::Time_t targetTime, bool debug );

    size_t numberOfQueries() const;

    const std::vector<std::string>& history() const;

    const Binary& getBuffer() const;
    size_t getBufferSize() const;

    const Binary& getStreamerInfo() const;

    void setRecordInfo(const std::string &recName, const std::string &recLabel) { m_recName = recName; m_recLabel = recLabel; }

    const std::string recName () const { return m_recName;  }
    const std::string recLabel() const { return m_recLabel; }

  private:
  
    struct pimpl {
      cond::Iov_t current;
      std::vector<std::string> history;
    };
    
    Session m_session;
    IOVProxy m_iov;
    boost::shared_ptr<pimpl> m_data;
  
    Binary m_buffer;
    Binary m_streamerInfo;

    std::string m_recName;
    std::string m_recLabel;

  }; //  end class UntypedPayloadProxy

  class TestGTPerf : public cond::Utilities {
    public:
      TestGTPerf();
      int execute() override;
  }; // end class TestGTLoad
  
} // end namespace cond

cond::ConnectionPoolWrapper::ConnectionPoolWrapper( int authenticationSystem, const std::string& authenticationPath, bool debug ){
  connPool.setAuthenticationSystem( authenticationSystem );
  if( !authenticationPath.empty() ) connPool.setAuthenticationPath( authenticationPath );
  if( debug ) connPool.setMessageVerbosity( coral::Debug );
  connPool.configure();
}

cond::Session cond::ConnectionPoolWrapper::createSession( const std::string& connectionString ){
  Session s;
  {
    boost::mutex::scoped_lock slock( lock );
    s = connPool.createSession( connectionString );
  }
  return s;
}

cond::UntypedPayloadProxy::UntypedPayloadProxy():
  m_session(),
  m_iov(),
  m_data(), 
  m_buffer() {
  m_data.reset( new pimpl );
  m_data->current.clear();
}

cond::UntypedPayloadProxy::UntypedPayloadProxy( const UntypedPayloadProxy& rhs ):
  m_session( rhs.m_session ),
  m_iov( rhs.m_iov ),
  m_data( rhs.m_data ),
  m_buffer( rhs.m_buffer ){
}

cond::UntypedPayloadProxy& cond::UntypedPayloadProxy::operator=( const cond::UntypedPayloadProxy& rhs ){
  m_session = rhs.m_session;
  m_iov = rhs.m_iov;
  m_data = rhs.m_data;
  m_buffer = rhs.m_buffer;
  return *this;
}

void cond::UntypedPayloadProxy::init( Session session ){
  m_session = session;
  reset();
}

void cond::UntypedPayloadProxy::load( const std::string& tag ){
  m_data->current.clear();
  m_session.transaction().start();
  m_iov = m_session.readIov( tag );
  m_session.transaction().commit();
}

void cond::UntypedPayloadProxy::reload(){
  std::string tag = m_iov.tag();
  load( tag );
}

void cond::UntypedPayloadProxy::reset(){
  m_iov.reset();
  m_data->current.clear();
}

void cond::UntypedPayloadProxy::disconnect(){
  m_session.close();
}

std::string cond::UntypedPayloadProxy::tag() const {
  return m_iov.tag();
}

cond::TimeType cond::UntypedPayloadProxy::timeType() const {
  return m_iov.timeType();
}

std::string cond::UntypedPayloadProxy::payloadType() const {
  return m_iov.payloadObjectType();
}

bool cond::UntypedPayloadProxy::get( cond::Time_t targetTime, bool debug ){
  bool loaded = false;

  //  check if the current iov loaded is the good one...
  if( targetTime < m_data->current.since || targetTime >= m_data->current.till ){

    // a new payload is required!
    if( debug )std::cout <<" Searching tag "<<m_iov.tag()<<" for a valid payload for time="<<targetTime<<std::endl;
    m_session.transaction().start();
    auto iIov = m_iov.find( targetTime );
    if(iIov == m_iov.end() ) cond::throwException(std::string("Tag ")+m_iov.tag()+": No iov available for the target time:"+std::to_string(targetTime),"UntypedPayloadProxy::get");
    m_data->current = *iIov;

    std::string payloadType(""); 
    loaded = m_session.fetchPayloadData( m_data->current.payloadId, payloadType, m_buffer, m_streamerInfo );
    m_session.transaction().commit();
    
    if( !loaded ){
      std::cout <<"ERROR: payload with id "<<m_data->current.payloadId<<" could not be loaded."<<std::endl;
    } else {
      if( debug ) std::cout <<"Loaded payload of type \""<< payloadType <<"\" (" << m_buffer.size() << " bytes)"<<std::endl;  
    }
    // check if hash is correct:
    cond::Hash localHash = cond::persistency::makeHash( payloadType, m_buffer );
    if ( localHash != m_data->current.payloadId ) {
      std::cout <<"ERROR: payload of type " << payloadType << " with id " << m_data->current.payloadId << " in DB has wrong local hash: " << localHash << std::endl;
    }
  }
  return loaded;
}

size_t cond::UntypedPayloadProxy::numberOfQueries() const {
  return m_iov.numberOfQueries();
}

const std::vector<std::string>& cond::UntypedPayloadProxy::history() const {
  return m_data->history;
}

size_t cond::UntypedPayloadProxy::getBufferSize() const {
    return m_buffer.size();
}

const cond::Binary& cond::UntypedPayloadProxy::getBuffer() const {
    return m_buffer;
}

const cond::Binary& cond::UntypedPayloadProxy::getStreamerInfo() const {
    return m_streamerInfo;
}

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
    if (fetchTime.empty()) {
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
    if (deserTime.empty()) {
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

cond::TestGTPerf::TestGTPerf():
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
  addOption<int>("n_fetch","f","number of threads to load payloads (default=1)");
  addOption<int>("n_deser","d","number of threads do deserialize payloads (default=1)");
}


// thread helpers

// global counter for dummy thread measurements:
volatile int fooGlobal = 0;

class FetchWorker {
private:
  cond::ConnectionPoolWrapper& connectionPool;
  std::string connectionString;
  cond::UntypedPayloadProxy *p;
  std::map<std::string,size_t> *requests;
  cond::Time_t runSel;
  cond::Time_t lumiSel;
  cond::Time_t tsSel;

  boost::mutex my_lock;
public:
  FetchWorker( cond::ConnectionPoolWrapper& connPool, 
	       const std::string& connString,
	       cond::UntypedPayloadProxy *pIn, 
	       std::map<std::string,size_t> *reqIn, 
	       const cond::Time_t &run, 
	       const cond::Time_t &lumi, 
	       const cond::Time_t &ts) : 
    connectionPool( connPool ),
    connectionString( connString ),
    p(pIn), 
    requests(reqIn),
    runSel(run), lumiSel(lumi), tsSel(ts)
  {
  }

  void run() { runReal() ; }

  void runFake() {
    fooGlobal++;
  }
  void runReal() {
    bool debug  = false;
    bool loaded = false;
    cond::time::TimeType ttype = p->timeType();
    auto r = requests->find( p->tag() );
    cond::Session s;
    try{
      s = connectionPool.createSession( connectionString );
      p->init( s ); 
      p->reload();
      if( ttype==cond::runnumber ){
	p->get( runSel, debug );	
	boost::mutex::scoped_lock slock( my_lock );
	r->second++;
      } else if( ttype==cond::lumiid ){
	p->get( lumiSel, debug );
	boost::mutex::scoped_lock slock( my_lock );
	r->second++;
      } else if( ttype==cond::timestamp){
	p->get( tsSel, debug );
	boost::mutex my_lock;
	r->second++;
      } else {
	std::cout <<"WARNING: iov request on tag "<<p->tag()<<" (timeType="<<cond::time::timeTypeName(p->timeType())<<") has been skipped."<<std::endl;
      }
      s.close();
      //-ap:  not thread-safe!  timex.fetchInt(p->getBufferSize()); // keep track of time vs. size
    } catch ( const cond::Exception& e ){
      std::cout <<"ERROR:"<<e.what()<<std::endl;
    }
  }
};


class DeserialWorker {
private:  
  cond::UntypedPayloadProxy *p;
  std::shared_ptr<void> payload;
  boost::mutex my_lock;

public:
  DeserialWorker(cond::UntypedPayloadProxy *pIn, std::shared_ptr<void> &plIn) : p(pIn), payload(plIn) {}

  void run() { runReal(); }

  void runFake() {    
    fooGlobal++;
  }
  void runReal() {
    std::shared_ptr<void> payloadPtr;
    std::string payloadTypeName =  p->payloadType();
    const cond::Binary &buffer = p->getBuffer();
    const cond::Binary &streamerInfo = p->getStreamerInfo();
  
    auto result = std::make_unique<std::pair< std::string, std::shared_ptr<void> > >(cond::persistency::fetchOne( payloadTypeName, buffer, streamerInfo, payloadPtr ));
    payload = result->second;

    return;
  }
};

template <typename T> struct invoker {
  void operator()(T& it) const {it->run();}
};

int cond::TestGTPerf::execute(){

  std::string gtag = getOptionValue<std::string>("globaltag");
  bool debug = hasDebug();
  std::string connect = getOptionValue<std::string>("connect");
  bool verbose = hasOptionValue("verbose");

  int nThrF = getOptionValue<int>("n_fetch");
  int nThrD = getOptionValue<int>("n_deser");
  std::cout << "\n++> going to use " << nThrF << " threads for loading, " << nThrD << " threads for deserialization. \n" << std::endl;

  std::string serType = "unknown";
  if ( connect.find("CMS_CONDITIONS") != -1 ) {
    serType = "ROOT-5";
  } else if (connect.find("CMS_TEST_CONDITIONS") != -1 ) {
    serType = "boost";
  }

  Time_t startRun= 150005;
  if(hasOptionValue("start_run")) startRun = getOptionValue<Time_t>("start_run");
  Time_t startTs= 5800013687234232320;
  if(hasOptionValue("start_ts")) startTs = getOptionValue<Time_t>("start_ts");
  Time_t startLumi= 908900979179966;
  if(hasOptionValue("start_lumi")) startLumi = getOptionValue<Time_t>("start_lumi");

  std::string authPath("");
  if( hasOptionValue("authPath")) authPath = getOptionValue<std::string>("authPath");

  initializePluginManager();

  Timer timex(serType);

  ConnectionPoolWrapper connPool( 1, authPath, hasDebug() );
  Session session = connPool.createSession( connect );
  session.transaction().start();
  
  std::cout <<"Loading Global Tag "<<gtag<<std::endl;
  GTProxy gt = session.readGlobalTag( gtag );

  session.transaction().commit();

  std::cout <<"Loading "<<gt.size()<<" tags..."<<std::endl;
  std::vector<UntypedPayloadProxy *> proxies;
  std::map<std::string,size_t> requests;
  size_t nt = 0;
  for( auto t: gt ){
    nt++;
    UntypedPayloadProxy * p = new UntypedPayloadProxy;
    p->init( session );
    try{
      p->load( t.tagName() );
      if (nThrF == 1) { // detailed info only needed in single-threaded mode to get the types/names
	p->setRecordInfo( t.recordName(), t.recordLabel() );
      }
      proxies.push_back( p );
      requests.insert( std::make_pair( t.tagName(), 0 ) );
    } catch ( const cond::Exception& e ){
      std::cout <<"ERROR: "<<e.what()<<std::endl;
    }
  }
  std::cout << proxies.size() << " tags successfully loaded." << std::endl;
  timex.interval("loading iovs");
  
  Time_t run = startRun;
  Time_t lumi = startLumi;
  Time_t ts = startTs;

  if (nThrF > 1) session.transaction().commit();

  tbb::task_scheduler_init init( nThrF );
  std::vector<std::shared_ptr<FetchWorker> > tasks;

  std::string payloadTypeName;
  for( auto p: proxies ){
      payloadTypeName = p->payloadType();
      // ignore problematic ones for now
      if ( (payloadTypeName == "SiPixelGainCalibrationOffline")  // 2 * 133 MB !!!
	   ) { 
	std::cout << "WARNING: Ignoring problematic payload of type " << payloadTypeName << std::endl;
	continue;
      }

      if (nThrF > 1) {
        auto fw = std::make_shared<FetchWorker>(connPool, connect, p, (std::map<std::string,size_t> *) &requests,
							    run, lumi, ts);
	tasks.push_back(fw);
      } else {
	bool loaded = false;
	time::TimeType ttype = p->timeType();
	auto r = requests.find( p->tag() );
	try{
	  if( ttype==runnumber ){
	    p->get( run, hasDebug() );	
	    r->second++;
	  } else if( ttype==lumiid ){
	    p->get( lumi, hasDebug() );
	    r->second++;
	  } else if( ttype==timestamp){
	    p->get( ts, hasDebug() );
	    r->second++;
	  } else {
	    std::cout <<"WARNING: iov request on tag "<<p->tag()<<" (timeType="<<time::timeTypeName(p->timeType())<<") has been skipped."<<std::endl;
	  }
	  timex.fetchInt(p->getBufferSize()); // keep track of time vs. size
	} catch ( const cond::Exception& e ){
	  std::cout <<"ERROR:"<<e.what()<<std::endl;
	}
      } // end else (single thread)
  }

  tbb::parallel_for_each(tasks.begin(),tasks.end(),invoker<std::shared_ptr<FetchWorker> >() );

  std::cout << "global counter : " << fooGlobal << std::endl;

  if (nThrF == 1) session.transaction().commit();
  // session.transaction().commit();

  timex.interval("loading payloads");

  size_t totBufSize = 0;
  for( auto p: proxies ){
      totBufSize += p->getBufferSize();
  }
  std::cout << "++> total buffer size used : " << totBufSize << std::endl;

  std::vector<std::shared_ptr<void> > payloads;
  payloads.resize(400); //-todo: check we don't have more payloads than that !!

  std::shared_ptr<void> payloadPtr;

  tbb::task_scheduler_init initD( nThrD );
  std::vector<std::shared_ptr<DeserialWorker> > tasksD;

  timex.interval("setup deserialization");

  int nEmpty = 0;
  int nBig = 0;
  int index = 0;
  for( auto p: proxies ){

///     if ( p->getBufferSize() == 0 ) { // nothing to do for these ... 
///       std::cout << "empty buffer found for " << p->payloadType() << std::endl;
///       nEmpty++;
///       continue;
///     }

    payloadTypeName = p->payloadType();

    // ignore problematic ones for now
    if ( (payloadTypeName == "SiPixelGainCalibrationForHLT")
	 or (payloadTypeName == "SiPixelGainCalibrationOffline")  // 2 * 133 MB !!!
	 or (payloadTypeName == "DTKeyedConfig") 
	 or (payloadTypeName == "std::vector<unsigned long long>") 
	 or (payloadTypeName == "  AlignmentSurfaceDeformations")
	 // only in root for now:
	 or (payloadTypeName == "PhysicsTools::Calibration::MVAComputerContainer")
	 or (payloadTypeName == "PhysicsTools::Calibration::MVAComputerContainer")
	 or (payloadTypeName == "PhysicsTools::Calibration::MVAComputerContainer")
	 or (payloadTypeName == "PhysicsTools::Calibration::MVAComputerContainer")
	 ) { 
      std::cout << "INFO: Ignoring payload of type " << payloadTypeName << std::endl;
      continue;
    }
    
    if (nThrD > 1) {
      auto dw = std::make_shared<DeserialWorker>(p, payloads[index]);
      tasksD.push_back(dw); 
    } else { // single tread only
       try {
	 std::pair<std::string, std::shared_ptr<void> > result = fetchOne( payloadTypeName, p->getBuffer(), p->getStreamerInfo(), payloadPtr);
           payloads.push_back(result.second);
       } catch ( const cond::Exception& e ){
           std::cout << "\nERROR (cond): " << e.what() << std::endl;
           std::cout << "for payload type name: " << payloadTypeName << std::endl;
       } catch ( const std::exception& e ){
           std::cout << "\nERROR (boost/std): " << e.what() << std::endl;
           std::cout << "for payload type name: " << payloadTypeName << std::endl;
       }
       timex.deserInt(p->getBufferSize()); // keep track of time vs. size
    } // single-thread
    index++; // increment index into payloads
  }
  std::cout << std::endl;

  tbb::parallel_for_each(tasksD.begin(),tasksD.end(),invoker<std::shared_ptr<DeserialWorker> >() );
 
  timex.interval("deserializing payloads");

  std::cout << "global counter : " << fooGlobal << std::endl;
  std::cout << "found   " << nEmpty << " empty payloads while deserialising " << std::endl;

  std::cout <<std::endl;
  std::cout <<"*** End of job."<<std::endl;
  std::cout <<"*** GT: "<<gtag<<" Tags:"<<gt.size()<<" Loaded:"<<proxies.size()<<std::endl;
  std::cout<<std::endl;
  for( auto p: proxies ){
    auto r = requests.find( p->tag() );
    if( verbose ){
      std::cout <<"*** Tag: "<<p->tag()<<" Requests processed:"<<r->second<<" Queries:"<< p->numberOfQueries() <<std::endl;
      const std::vector<std::string>& hist = p->history();
      for( auto e: p->history() ) std::cout <<"    "<<e<<std::endl;
    }
  }

  // only for igprof checking of live mem:
  // ::exit(0);

  timex.interval("postprocessing ... ");
  timex.showIntervals();
  
  if ( nThrF == 1) {
    std::ofstream ofs("fetchInfo.txt");
    timex.showFetchInfo(ofs);
    std::ofstream ofs2("sizeInfo.txt");
    for ( auto p: proxies ) {
      ofs2 << p->payloadType() << "[" << p->recName() << ":" << p->recLabel() << "]" << " : " << p->getBufferSize() << std::endl;
    }
  }
  if ( nThrD == 1) {
    std::ofstream ofs1("deserializeInfo.txt");
    timex.showDeserInfo(ofs1);
  }
  
  return 0;
}

// ================================================================================

int main( int argc, char** argv ){

  // usage: conddb_test_gt_perf -g START70_V1 -n 1 -c oracle://cms_orcoff_prep/CMS_CONDITIONS --n_fetch 1 --n_deser 1 2>&1 | tee run.log
  // usage: conddb_test_gt_perf -g START70_V1 -n 1 -c oracle://cms_orcoff_prep/CMS_TEST_CONDITIONS --n_fetch 2 --n_deser 8 2>&1 | tee run_2f_8d.log
    
  cond::TestGTPerf test;
  return test.run(argc,argv);
}

