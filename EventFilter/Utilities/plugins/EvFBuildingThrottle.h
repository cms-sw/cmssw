#ifndef EVENTFILTER_UTILTIES_PLUGINS_EVFBuildingThrottle
#define EVENTFILTER_UTILTIES_PLUGINS_EVFBuildingThrottle

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "EvFDaqDirector.h"

#include <sys/statvfs.h>

#include "boost/thread/thread.hpp"


namespace evf{
  class EvFBuildingThrottle 
    {
    public:
      enum Directory { mInvalid = 0, mBase, mBU, mSM, mMon, mWhat, 
		       mCOUNT}; 
      explicit EvFBuildingThrottle( const edm::ParameterSet &pset, 
				    edm::ActivityRegistry& reg ) 
	: highWaterMark_(pset.getUntrackedParameter<double>("highWaterMark",0.8))
	, lowWaterMark_(pset.getUntrackedParameter<double>("lowWaterMark",0.5))
	, m_stoprequest(false)
	, whatToThrottleOn_(Directory(pset.getUntrackedParameter<int>("dirCode",mBase)))
	, throttled_(false)
	, sleep_( pset.getUntrackedParameter<unsigned int>("sleepmSecs",1000))
      {
	reg.watchPreGlobalBeginRun(this,&EvFBuildingThrottle::preBeginRun);  
	reg.watchPostGlobalEndRun(this,&EvFBuildingThrottle::postEndRun);  
	reg.watchPreGlobalBeginLumi(this,&EvFBuildingThrottle::preBeginLumi);
      }
      ~EvFBuildingThrottle(){}
//      void preBeginRun(edm::RunID const& id, edm::Timestamp const& ts){
      void preBeginRun(edm::GlobalContext const& gc){
	//obtain directory to stat on
	switch(whatToThrottleOn_){
	case mInvalid:
	  //do nothing
	  break;
	case mBase:
	  baseDir_ = edm::Service<EvFDaqDirector>()->baseDir();
	  break;
	case mBU:
	  baseDir_ = edm::Service<EvFDaqDirector>()->buBaseDir();
	  break;
	case mSM:
	  baseDir_ = edm::Service<EvFDaqDirector>()->smBaseDir();
	  break;
	case mMon:
	  baseDir_ = edm::Service<EvFDaqDirector>()->monitorBaseDir();
	  break;
	default:
	  baseDir_ = edm::Service<EvFDaqDirector>()->baseDir();
	}
	start();
      }
      void postBeginRun(edm::GlobalContext const& gc){
      }

      void postEndRun(edm::GlobalContext const& gc){
	stop();
      }
      void preBeginLumi(edm::GlobalContext const& gc){
	lock_.lock();
	lock_.unlock();
      }
      bool throttled() const {return throttled_;}
    private:
      void dowork(){
	edm::ServiceRegistry::Operate operate(token_);
	struct statvfs buf;
	while(!m_stoprequest){
	  int retval = statvfs(baseDir_.c_str(),&buf);
	  if(retval != 0){
	    std::cout << " building throttle - unable to stat " << baseDir_ << std::endl;
	    m_stoprequest=true;
	    continue;
	  }
	  double fraction = 1.-float(buf.f_bfree*buf.f_bsize)/float(buf.f_blocks*buf.f_frsize);
	  bool highwater_ = fraction>highWaterMark_;
	  bool lowwater_ = fraction<lowWaterMark_;
	  if(highwater_ && !throttled_){ lock_.lock(); throttled_ = true;std::cout << ">>>>throttling on " << std::endl;}
	  if(lowwater_ && throttled_){ lock_.unlock(); throttled_ = false;}
	  std::cout << " building throttle on " << baseDir_ << " is " << fraction*100 << " %full " << std::endl;
	  edm::Service<EvFDaqDirector>()->writeDiskAndThrottleStat(fraction,highwater_,lowwater_);
	  ::usleep(sleep_*1000);
	}
      }
      void start(){
	assert(!m_thread);
	token_ = edm::ServiceRegistry::instance().presentToken();
	m_thread = boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(&EvFBuildingThrottle::dowork,this)));
	std::cout << "throttle thread started - throttle on " <<  whatToThrottleOn_ << std::endl;
      }
      void stop(){
	assert(m_thread);
	m_stoprequest=true;
	m_thread->join();
      }
      
      double highWaterMark_;
      double lowWaterMark_;
      volatile bool m_stoprequest;
      boost::shared_ptr<boost::thread> m_thread;
      boost::mutex lock_;
      std::string baseDir_;
      Directory whatToThrottleOn_;
      edm::ServiceToken token_;
      bool throttled_;
      unsigned int sleep_;
  };
}

#endif

