#ifndef EVF_FASTMONITORINGTHREAD
#define EVF_FASTMONITORINGTHREAD

#include "EventFilter/Utilities/interface/FastMonitor.h"

#include "boost/thread/thread.hpp"

#include <iostream>
#include <vector>



using namespace jsoncollector;

namespace evf{

  class FastMonitoringService;

  class FastMonitoringThread{
  public:
    // a copy of the Framework/EventProcessor states 
    enum Macrostate { sInit = 0, sJobReady, sRunGiven, sRunning, sStopping,
		      sShuttingDown, sDone, sJobEnded, sError, sErrorEnded, sEnd, sInvalid,MCOUNT}; 
    struct StreamMonitorData
    {
      //fastpath global monitorables
      IntJ fastMacrostateJ_;
      DoubleJ fastThroughputJ_;
      DoubleJ fastAvgLeadTimeJ_;
      IntJ fastFilesProcessedJ_;

      unsigned int varIndexThrougput_;

      //per stream
      std::vector<unsigned int> microstateDecoded_;
      std::vector<unsigned int> ministateDecoded_;
      std::vector<std::atomic<unsigned int>> processed_;

      //tracking luminosity of a stream
      std::vector<std::atomic<unsigned int>> streamLumi_;

      //N bins for histograms
      unsigned int macrostateBins_;
      unsigned int ministateBins_;
      unsigned int microstateBins_;


      //unsigned int prescaleindex_; // ditto

      //monitored for JSON

      boost::shared_ptr<FastMonitor> jsonMonitor_;//collector

      MonitorData() {

	fastMacrostateJ_ = FastMonitoringThread::sInit;
	DoubleJ fastThroughputJ_ = 0;
	DoubleJ fastAvgLeadTimeJ_ = 0;
	IntJ fastFilesProcessedJ_ = 0;
        fastMacrostateJ_.setName("Macrostate");
        fastThroughputJ_.setName("Throughput");
        fastAvgLeadTimeJ_.setName("AverageLeadTime");
	fastFilesProcessedJ_.setName("FilesProcessed");
      }

      //to be called after fast monitor is constructed
      void registerVariables(FastMonitor &fm, unsigned int nStreams) {
	//tell FM to track these global variables(for fast and slow monitoring)
        fm.registerGlobalMonitorable(&fastMacrostateJ_,true,&microstateBins_);
        varIndexThroughput_ = fm.registerGlobalMonitorable(&fastThroughputJ_,false);
        fm.registerGlobalMonitorable(&fastAvgLeadTimeJ_,false);
        fm.registerGlobalMonitorable(&fastFilesProcessedJ_,false);

	for (unsigned int i=0;i<nStreams;i++) {
	  processed_.push_back(0);
          streamLumi_.push_back(0);
	}
	
	ministateDecoded_.resize(nstreams);
	microstateDecoded_.resize(nstreams);

	fm.setNStreams(nStreams);
	//tell FM to track these int vectors
        fm.registerStreamMonitorableUIntVec("Ministate", &ministateDecoded_,true,0,&ministateBins_);//what is 0 for ?
        fm.registerStreamMonitorableUIntVec("Microstate",&microstateDecoded_,true,0,&microstateBins_);
        fm.registerStreamMonitorableUIntVecAtomic("Processed",&processed_,false,0);
	//provide vector with updated per stream lumis and let it finish initialization
	fm.commit(&streamLumi_);
      }
    };

    //constructor
    FastMonitoringThread() : m_stoprequest(false) {
    }

    void resetFastMonitor(std::string const& microStateDefPath) {
      m_data.jsonMonitor_.reset(new FastMonitor(microStateDefPath));
    }

    void start(void (FastMonitoringService::*fp)(),FastMonitoringService *cp){
      assert(!m_thread);
      m_thread = boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(fp,cp)));
    }
    void stop(){
      assert(m_thread);
      m_stoprequest=true;
      m_thread->join();
    }

  private:

    volatile bool m_stoprequest;
    boost::shared_ptr<boost::thread> m_thread;
    MonitorData m_data;
    boost::mutex monlock_;

    std::auto_ptr<FastMonitor> jsonMonitor_;

    friend class FastMonitoringService;
  };
} //end namespace evf
#endif
