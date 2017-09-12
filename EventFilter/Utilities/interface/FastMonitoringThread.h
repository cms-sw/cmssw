#ifndef EVF_FASTMONITORINGTHREAD
#define EVF_FASTMONITORINGTHREAD

#include "EventFilter/Utilities/interface/FastMonitor.h"

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>


namespace evf{

  class FastMonitoringService;

  class FastMonitoringThread{
  public:
    // a copy of the Framework/EventProcessor states 
    enum Macrostate { sInit = 0, sJobReady, sRunGiven, sRunning, sStopping,
		      sShuttingDown, sDone, sJobEnded, sError, sErrorEnded, sEnd, sInvalid,MCOUNT}; 

    enum InputState { inIgnore = 0, inInit, inWaitInput, inNewLumi, inNewLumiBusyEndingLS, inNewLumiIdleEndingLS, inRunEnd, inProcessingFile, inWaitChunk , inChunkReceived,
                      inChecksumEvent, inCachedEvent, inReadEvent, inReadCleanup, inNoRequest, inNoRequestWithIdleThreads,
                      inNoRequestWithGlobalEoL, inNoRequestWithEoLThreads,
                      //supervisor thread and worker threads state
                      inSupFileLimit, inSupWaitFreeChunk, inSupWaitFreeChunkCopying, inSupWaitFreeThread, inSupWaitFreeThreadCopying, inSupBusy, inSupLockPolling,
                      inSupLockPollingCopying,inSupNoFile,inSupNewFile,inSupNewFileWaitThreadCopying,inSupNewFileWaitThread,
                      inSupNewFileWaitChunkCopying,inSupNewFileWaitChunk,
                      //combined with inWaitInput
                      inWaitInput_fileLimit,inWaitInput_waitFreeChunk,inWaitInput_waitFreeChunkCopying,inWaitInput_waitFreeThread,inWaitInput_waitFreeThreadCopying,
                      inWaitInput_busy,inWaitInput_lockPolling,inWaitInput_lockPollingCopying,inWaitInput_runEnd,
                      inWaitInput_noFile,inWaitInput_newFile,inWaitInput_newFileWaitThreadCopying,inWaitInput_newFileWaitThread,
                      inWaitInput_newFileWaitChunkCopying,inWaitInput_newFileWaitChunk,
                      //combined with inWaitChunk
                      inWaitChunk_fileLimit,inWaitChunk_waitFreeChunk,inWaitChunk_waitFreeChunkCopying,inWaitChunk_waitFreeThread,inWaitChunk_waitFreeThreadCopying,
                      inWaitChunk_busy,inWaitChunk_lockPolling,inWaitChunk_lockPollingCopying,inWaitChunk_runEnd,
                      inWaitChunk_noFile,inWaitChunk_newFile,inWaitChunk_newFileWaitThreadCopying,inWaitChunk_newFileWaitThread,
                      inWaitChunk_newFileWaitChunkCopying,inWaitChunk_newFileWaitChunk,
                      inCOUNT}; 

    struct MonitorData
    {
      //fastpath global monitorables
      jsoncollector::IntJ fastMacrostateJ_;
      jsoncollector::DoubleJ fastThroughputJ_;
      jsoncollector::DoubleJ fastAvgLeadTimeJ_;
      jsoncollector::IntJ fastFilesProcessedJ_;
      jsoncollector::DoubleJ fastLockWaitJ_;
      jsoncollector::IntJ fastLockCountJ_;
      jsoncollector::IntJ fastEventsProcessedJ_;

      unsigned int varIndexThrougput_;

      //per stream
      std::vector<unsigned int> microstateEncoded_;
      std::vector<unsigned int> ministateEncoded_;
      std::vector<jsoncollector::AtomicMonUInt*> processed_;
      jsoncollector::IntJ fastPathProcessedJ_;
      std::vector<unsigned int> threadMicrostateEncoded_;
      std::vector<unsigned int> inputState_;

      //tracking luminosity of a stream
      std::vector<unsigned int> streamLumi_;

      //N bins for histograms
      unsigned int macrostateBins_;
      unsigned int ministateBins_;
      unsigned int microstateBins_;
      unsigned int inputstateBins_;

      //unsigned int prescaleindex_; // ditto

      MonitorData() {

	fastMacrostateJ_ = FastMonitoringThread::sInit;
	fastThroughputJ_ = 0;
	fastAvgLeadTimeJ_ = 0;
	fastFilesProcessedJ_ = 0;
        fastLockWaitJ_ = 0;
        fastLockCountJ_ = 0;
        fastMacrostateJ_.setName("Macrostate");
        fastThroughputJ_.setName("Throughput");
        fastAvgLeadTimeJ_.setName("AverageLeadTime");
	fastFilesProcessedJ_.setName("FilesProcessed");
	fastLockWaitJ_.setName("LockWaitUs");
	fastLockCountJ_.setName("LockCount");

        fastPathProcessedJ_ = 0;
        fastPathProcessedJ_.setName("Processed");
      }

      //to be called after fast monitor is constructed
      void registerVariables(jsoncollector::FastMonitor* fm, unsigned int nStreams, unsigned int nThreads) {
	//tell FM to track these global variables(for fast and slow monitoring)
        fm->registerGlobalMonitorable(&fastMacrostateJ_,true,&macrostateBins_);
        fm->registerGlobalMonitorable(&fastThroughputJ_,false);
        fm->registerGlobalMonitorable(&fastAvgLeadTimeJ_,false);
        fm->registerGlobalMonitorable(&fastFilesProcessedJ_,false);
        fm->registerGlobalMonitorable(&fastLockWaitJ_,false);
        fm->registerGlobalMonitorable(&fastLockCountJ_,false);

	for (unsigned int i=0;i<nStreams;i++) {
	 jsoncollector::AtomicMonUInt * p  = new jsoncollector::AtomicMonUInt;
	 *p=0;
   	  processed_.push_back(p);
          streamLumi_.push_back(0);
	}
	
	microstateEncoded_.resize(nStreams);
	ministateEncoded_.resize(nStreams);
	threadMicrostateEncoded_.resize(nThreads);
	inputState_.resize(nStreams);
        for (unsigned int j=0;j<inputState_.size();j++) inputState_[j]=0;

	//tell FM to track these int vectors
        fm->registerStreamMonitorableUIntVec("Ministate", &ministateEncoded_,true,&ministateBins_);

	if (nThreads<=nStreams)//no overlapping in module execution per stream
          fm->registerStreamMonitorableUIntVec("Microstate",&microstateEncoded_,true,&microstateBins_);
	else
	  fm->registerStreamMonitorableUIntVec("Microstate",&threadMicrostateEncoded_,true,&microstateBins_);

        fm->registerStreamMonitorableUIntVecAtomic("Processed",&processed_,false,0);

        //input source state tracking (not stream, but other than first item in vector is set to Ignore state) 
        fm->registerStreamMonitorableUIntVec("Inputstate",&inputState_,true,&inputstateBins_);

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
      jsonMonitor_.reset(new jsoncollector::FastMonitor(microStateDefPath,defGroup,false));
      if (fastMicroStateDefPath.size())
        jsonMonitor_->addFastPathDefinition(fastMicroStateDefPath,defGroup,false);
    }

    void start(void (FastMonitoringService::*fp)(),FastMonitoringService *cp){
      assert(!m_thread);
      m_thread = std::make_shared<std::thread>(fp,cp);
    }
    void stop(){
      assert(m_thread);
      m_stoprequest=true;
      m_thread->join();
    }

  private:

    std::atomic<bool> m_stoprequest;
    std::shared_ptr<std::thread> m_thread;
    MonitorData m_data;
    std::mutex monlock_;

    std::unique_ptr<jsoncollector::FastMonitor> jsonMonitor_;

    friend class FastMonitoringService;
  };
} //end namespace evf
#endif
