#ifndef EvFFastMonitoringService_H
#define EvFFastMonitoringService_H 1

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <filesystem>

#include "EventFilter/Utilities/interface/FastMonitoringService.h"

#include <string>
#include <vector>
#include <map>
#include <queue>
#include <sstream>
#include <unordered_map>
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_scheduler_observer.h"

/*Description
  this is an evolution of the MicroStateService intended to be run in standalone multi-threaded cmsRun jobs
  A legenda for use by the monitoring process in the DAQ needs to be generated at beginJob (when first available). 
  We try to spare CPU time in the monitoring by avoiding even a single string lookup and using the 
  moduledesc pointer to key into the map instead and no string or string pointers are used for the microstates.
  Only a pointer value is stored using relaxed ordering at the time of module execution  which is fast.
  At snapshot time only (every few seconds) we do the map lookup to produce snapshot.
  The general counters and status variables (event number, number of processed events, number of passed and stored 
  events, luminosity section etc.) are also monitored here.
*/

class FedRawDataInputSource;
class DAQSource;

namespace edm {
  class ConfigurationDescriptions;
}

namespace evf {

  template <typename T>
  struct ContainableAtomic;
  class FastMonitoringThread;
  class ConcurrencyTracker;

  namespace FastMonState {

    enum Microstate {
      mInvalid = 0,
      mIdle,
      mFwkOvhSrc,
      mFwkOvhMod,
      mFwkEoL,
      mInput,
      mDqm,
      mBoL,
      mEoL,
      mGlobEoL,
      mFwk,
      mIdleSource,
      mEvent,
      mIgnore,
      mCOUNT,
    };

    enum Macrostate {
      sInit = 0,
      sJobReady,
      sRunGiven,
      sRunning,
      sStopping,
      sShuttingDown,
      sDone,
      sJobEnded,
      sError,
      sErrorEnded,
      sEnd,
      sInvalid,
      MCOUNT
    };

    enum InputState : short {
      inIgnore = 0,
      inInit,
      inWaitInput,
      inNewLumi,
      inNewLumiBusyEndingLS,
      inNewLumiIdleEndingLS,
      inRunEnd,
      inProcessingFile,
      inWaitChunk,
      inChunkReceived,
      inChecksumEvent,
      inCachedEvent,
      inReadEvent,
      inReadCleanup,
      inNoRequest,
      inNoRequestWithIdleThreads,
      inNoRequestWithGlobalEoL,
      inNoRequestWithEoLThreads,
      //supervisor thread and worker threads state
      inSupFileLimit,
      inSupWaitFreeChunk,
      inSupWaitFreeChunkCopying,
      inSupWaitFreeThread,
      inSupWaitFreeThreadCopying,
      inSupBusy,
      inSupLockPolling,
      inSupLockPollingCopying,
      inSupNoFile,
      inSupNewFile,
      inSupNewFileWaitThreadCopying,
      inSupNewFileWaitThread,
      inSupNewFileWaitChunkCopying,
      inSupNewFileWaitChunk,
      //combined with inWaitInput
      inWaitInput_fileLimit,
      inWaitInput_waitFreeChunk,
      inWaitInput_waitFreeChunkCopying,
      inWaitInput_waitFreeThread,
      inWaitInput_waitFreeThreadCopying,
      inWaitInput_busy,
      inWaitInput_lockPolling,
      inWaitInput_lockPollingCopying,
      inWaitInput_runEnd,
      inWaitInput_noFile,
      inWaitInput_newFile,
      inWaitInput_newFileWaitThreadCopying,
      inWaitInput_newFileWaitThread,
      inWaitInput_newFileWaitChunkCopying,
      inWaitInput_newFileWaitChunk,
      //combined with inWaitChunk
      inWaitChunk_fileLimit,
      inWaitChunk_waitFreeChunk,
      inWaitChunk_waitFreeChunkCopying,
      inWaitChunk_waitFreeThread,
      inWaitChunk_waitFreeThreadCopying,
      inWaitChunk_busy,
      inWaitChunk_lockPolling,
      inWaitChunk_lockPollingCopying,
      inWaitChunk_runEnd,
      inWaitChunk_noFile,
      inWaitChunk_newFile,
      inWaitChunk_newFileWaitThreadCopying,
      inWaitChunk_newFileWaitThread,
      inWaitChunk_newFileWaitChunkCopying,
      inWaitChunk_newFileWaitChunk,
      inSupThrottled,
      inThrottled,
      inCOUNT
    };
  }  // namespace FastMonState

  constexpr int nSpecialModules = FastMonState::mCOUNT;
  //reserve output module space
  constexpr int nReservedModules = 128;

  class FastMonitoringService {
  public:
    // the names of the states - some of them are never reached in an online app
    static const edm::ModuleDescription specialMicroStateNames[FastMonState::mCOUNT];
    static const std::string macroStateNames[FastMonState::MCOUNT];
    static const std::string inputStateNames[FastMonState::inCOUNT];
    // Reserved names for microstates
    FastMonitoringService(const edm::ParameterSet&, edm::ActivityRegistry&);
    ~FastMonitoringService();
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    std::string makeModuleLegendaJson();
    std::string makeInputLegendaJson();

    void preallocate(edm::service::SystemBounds const&);
    void jobFailure();
    void preBeginJob(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const& pc);

    void preModuleBeginJob(edm::ModuleDescription const&);
    void postBeginJob();
    void postEndJob();

    void postGlobalBeginRun(edm::GlobalContext const&);
    void preGlobalBeginLumi(edm::GlobalContext const&);
    void preGlobalEndLumi(edm::GlobalContext const&);
    void postGlobalEndLumi(edm::GlobalContext const&);

    void preStreamBeginLumi(edm::StreamContext const&);
    void postStreamBeginLumi(edm::StreamContext const&);
    void preStreamEndLumi(edm::StreamContext const&);
    void postStreamEndLumi(edm::StreamContext const&);
    void prePathEvent(edm::StreamContext const&, edm::PathContext const&);
    void preEvent(edm::StreamContext const&);
    void postEvent(edm::StreamContext const&);
    void preSourceEvent(edm::StreamID);
    void postSourceEvent(edm::StreamID);
    void preModuleEventAcquire(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void postModuleEventAcquire(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void preStreamEarlyTermination(edm::StreamContext const&, edm::TerminationOrigin);
    void preGlobalEarlyTermination(edm::GlobalContext const&, edm::TerminationOrigin);
    void preSourceEarlyTermination(edm::TerminationOrigin);
    void setExceptionDetected(unsigned int ls);

    void accumulateFileSize(unsigned int lumi, unsigned long fileSize);
    void startedLookingForFile();
    void stoppedLookingForFile(unsigned int lumi);
    void reportLockWait(unsigned int ls, double waitTime, unsigned int lockCount);
    unsigned int getEventsProcessedForLumi(unsigned int lumi, bool* abortFlag = nullptr);
    bool getAbortFlagForLumi(unsigned int lumi);
    bool exceptionDetected() const;
    bool isExceptionOnData(unsigned int ls);
    bool shouldWriteFiles(unsigned int lumi, unsigned int* proc = nullptr) {
      unsigned int processed = getEventsProcessedForLumi(lumi);
      if (proc)
        *proc = processed;
      return !getAbortFlagForLumi(lumi);
    }
    std::string getRunDirName() const { return runDirectory_.stem().string(); }
    void setInputSource(FedRawDataInputSource* inputSource) { inputSource_ = inputSource; }
    void setInputSource(DAQSource* inputSource) { daqInputSource_ = inputSource; }
    void setInState(FastMonState::InputState inputState) { inputState_ = inputState; }
    void setInStateSup(FastMonState::InputState inputState) { inputSupervisorState_ = inputState; }
    //available for other modules
    void setTMicrostate(FastMonState::Microstate m);

    static unsigned int getTID() { return tbb::this_task_arena::current_thread_index(); }

  private:
    void doSnapshot(const unsigned int ls, const bool isGlobalEOL);

    void snapshotRunner();

    static unsigned int getSID(edm::StreamContext const& sc) { return sc.streamID().value(); }

    static unsigned int getSID(edm::StreamID const& sid) { return sid.value(); }

    //the actual monitoring thread is held by a separate class object for ease of maintenance
    std::unique_ptr<FastMonitoringThread> fmt_;
    std::unique_ptr<ConcurrencyTracker> ct_;
    //Encoding encModule_;
    //std::vector<Encoding> encPath_;
    FedRawDataInputSource* inputSource_ = nullptr;
    DAQSource* daqInputSource_ = nullptr;
    std::atomic<FastMonState::InputState> inputState_{FastMonState::InputState::inInit};
    std::atomic<FastMonState::InputState> inputSupervisorState_{FastMonState::InputState::inInit};

    unsigned int nStreams_ = 0;
    unsigned int nMonThreads_ = 0;
    unsigned int nThreads_ = 0;
    bool tbbMonitoringMode_;
    bool tbbConcurrencyTracker_;
    int sleepTime_;
    unsigned int fastMonIntervals_;
    unsigned int snapCounter_ = 0;
    std::string microstateDefPath_, fastMicrostateDefPath_;
    std::string fastName_, fastPath_;

    //variables that are used by/monitored by FastMonitoringThread / FastMonitor

    std::map<unsigned int, timeval> lumiStartTime_;  //needed for multiplexed begin/end lumis
    timeval fileLookStart_, fileLookStop_;           //this could also be calculated in the input source

    unsigned int lastGlobalLumi_;
    std::atomic<bool> isInitTransition_;
    unsigned int lumiFromSource_;

    //variables measuring source statistics (global)
    //unordered_map is not used because of very few elements stored concurrently
    std::map<unsigned int, double> avgLeadTime_;
    std::map<unsigned int, unsigned int> filesProcessedDuringLumi_;
    //helpers for source statistics:
    std::map<unsigned int, unsigned long> accuSize_;
    std::vector<double> leadTimes_;
    std::map<unsigned int, std::pair<double, unsigned int>> lockStatsDuringLumi_;

    //for output module
    std::map<unsigned int, std::pair<unsigned int, bool>> processedEventsPerLumi_;

    //flag used to block EOL until event count is picked up by caches (not certain that this is really an issue)
    //to disable this behavior, set #ATOMIC_LEVEL 0 or 1 in DataPoint.h
    std::vector<std::atomic<bool>*> streamCounterUpdating_;

    std::filesystem::path workingDirectory_, runDirectory_;

    bool threadIDAvailable_ = false;

    std::atomic<unsigned long> totalEventsProcessed_;

    std::string moduleLegendFile_;
    std::string moduleLegendFileJson_;
    std::string inputLegendFileJson_;
    unsigned int nOutputModules_ = 0;

    std::atomic<bool> monInit_;
    bool exception_detected_ = false;
    std::atomic<bool> has_source_exception_ = false;
    std::atomic<bool> has_data_exception_ = false;
    std::vector<unsigned int> exceptionInLS_;

    //per stream
    std::vector<ContainableAtomic<const void*>> microstate_;
    std::vector<ContainableAtomic<unsigned char>> microstateAcqFlag_;
    //per thread
    std::vector<ContainableAtomic<const void*>> tmicrostate_;
    std::vector<ContainableAtomic<unsigned char>> tmicrostateAcqFlag_;

    bool verbose_ = false;
  };

}  // namespace evf

#endif
