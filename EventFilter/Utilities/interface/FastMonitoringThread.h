#ifndef EVF_FASTMONITORINGTHREAD
#define EVF_FASTMONITORINGTHREAD

#include "EventFilter/Utilities/interface/FastMonitor.h"

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

using namespace jsoncollector;

namespace evf{

  class FastMonitoringService;

  class FastMonitoringThread{
  public:
    // a copy of the Framework/EventProcessor states 
    enum Macrostate { sInit = 0, sJobReady, sRunGiven, sRunning, sStopping,
		      sShuttingDown, sDone, sJobEnded, sError, sErrorEnded, sEnd, sInvalid,MCOUNT}; 
    struct MonitorData
    {
      //fastpath global monitorables
      IntJ fastMacrostateJ_;
      DoubleJ fastThroughputJ_;
      DoubleJ fastAvgLeadTimeJ_;
      IntJ fastFilesProcessedJ_;

      unsigned int varIndexThrougput_;

      //per stream
      std::vector<unsigned int> microstateEncoded_;
      std::vector<unsigned int> ministateEncoded_;
      std::vector<AtomicMonUInt*> processed_;
      IntJ fastPathProcessedJ_;
      std::vector<unsigned int> threadMicrostateEncoded_;

      //tracking luminosity of a stream
      std::vector<unsigned int> streamLumi_;

      //N bins for histograms
      unsigned int macrostateBins_;
      unsigned int ministateBins_;
      unsigned int microstateBins_;

      //unsigned int prescaleindex_; // ditto

      MonitorData() {

	fastMacrostateJ_ = FastMonitoringThread::sInit;
	fastThroughputJ_ = 0;
	fastAvgLeadTimeJ_ = 0;
	fastFilesProcessedJ_ = 0;
        fastMacrostateJ_.setName("Macrostate");
        fastThroughputJ_.setName("Throughput");
        fastAvgLeadTimeJ_.setName("AverageLeadTime");
	fastFilesProcessedJ_.setName("FilesProcessed");

        fastPathProcessedJ_ = 0;
        fastPathProcessedJ_.setName("Processed");
      }

      //to be called after fast monitor is constructed
      void registerVariables(FastMonitor* fm, unsigned int nStreams, unsigned int nThreads) {
	//tell FM to track these global variables(for fast and slow monitoring)
        fm->registerGlobalMonitorable(&fastMacrostateJ_,true,&macrostateBins_);
        fm->registerGlobalMonitorable(&fastThroughputJ_,false);
        fm->registerGlobalMonitorable(&fastAvgLeadTimeJ_,false);
        fm->registerGlobalMonitorable(&fastFilesProcessedJ_,false);

	for (unsigned int i=0;i<nStreams;i++) {
	 AtomicMonUInt * p  = new AtomicMonUInt;
	 *p=0;
   	  processed_.push_back(p);
          streamLumi_.push_back(0);
	}
	
	microstateEncoded_.resize(nStreams);
	ministateEncoded_.resize(nStreams);
	threadMicrostateEncoded_.resize(nThreads);

	//tell FM to track these int vectors
        fm->registerStreamMonitorableUIntVec("Ministate", &ministateEncoded_,true,&ministateBins_);

	if (nThreads<=nStreams)//no overlapping in module execution per stream
          fm->registerStreamMonitorableUIntVec("Microstate",&microstateEncoded_,true,&microstateBins_);
	else
	  fm->registerStreamMonitorableUIntVec("Microstate",&threadMicrostateEncoded_,true,&microstateBins_);

        fm->registerStreamMonitorableUIntVecAtomic("Processed",&processed_,false,0);

        //global cumulative event counter is used for fast path
        fm->registerFastGlobalMonitorable(&fastPathProcessedJ_);

	//provide vector with updated per stream lumis and let it finish initialization
	fm->commit(&streamLumi_);
      }
    };

    //constructor
    FastMonitoringThread() : m_stoprequest(false) {
    }

    void resetFastMonitor(std::string const& microStateDefPath, std::string const& fastMicroStateDefPath) {
      std::string defGroup = "data";
      jsonMonitor_.reset(new FastMonitor(microStateDefPath,defGroup,false));
      if (fastMicroStateDefPath.size())
        jsonMonitor_->addFastPathDefinition(fastMicroStateDefPath,defGroup,false);
    }

    void start(void (FastMonitoringService::*fp)(),FastMonitoringService *cp){
      assert(!m_thread);
      m_thread = boost::shared_ptr<std::thread>(new std::thread(fp,cp));
    }
    void stop(){
      assert(m_thread);
      m_stoprequest=true;
      m_thread->join();
    }

  private:

    std::atomic<bool> m_stoprequest;
    boost::shared_ptr<std::thread> m_thread;
    MonitorData m_data;
    std::mutex monlock_;

    std::unique_ptr<FastMonitor> jsonMonitor_;

    friend class FastMonitoringService;
  };
} //end namespace evf
#endif
