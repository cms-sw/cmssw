#ifndef EvFFastMonitoringService_H
#define EvFFastMonitoringService_H 1

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <filesystem>

#include "EventFilter/Utilities/interface/MicroStateService.h"

#include <string>
#include <vector>
#include <map>
#include <queue>
#include <sstream>
#include <unordered_map>

/*Description
  this is an evolution of the MicroStateService intended to be run standalone in cmsRun or similar
  As such, it has to independently create a monitoring thread and run it in each forked process, which needs 
  to be arranged following the standard CMSSW procedure.
  A legenda for use by the monitoring process in the DAQ needs to be generated as soon as convenient - since 
  no access to the EventProcessor is granted, this needs to wait until after beginJob is executed.
  At the same time, we try to spare time in the monitoring by avoiding even a single string lookup and using the 
  moduledesc pointer to key into the map instead.
  As a bonus, we can now add to the monitored status the current path (and possibly associate modules to a path...)
  this intermediate info will be called "ministate" :D
  The general counters and status variables (event number, number of processed events, number of passed and stored 
  events, luminosity section etc. are also monitored here.

  NOTA BENE!!! with respect to the MicroStateService, no string or string pointers are used for the microstates.
  NOTA BENE!!! the state of the edm::EventProcessor cannot be monitored directly from within a service, so a 
  different solution must be identified for that (especially one needs to identify error states). 
  NOTA BENE!!! to keep backward compatibility with the MicroStateService, a common base class with abstract interface,
  exposing the single  method to be used by all other packages (except EventFilter/Processor, 
  which should continue to use the concrete class interface) will be defined 

*/
class FedRawDataInputSource;

namespace edm {
  class ConfigurationDescriptions;
}

namespace evf {

  class FastMonitoringThread;

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
      mCOUNT
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

    enum InputState {
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
      inCOUNT
    };
  }//namespace FastMonStates


  class FastMonitoringService : public MicroStateService {

  public:
    // the names of the states - some of them are never reached in an online app
    static const edm::ModuleDescription reservedMicroStateNames[FastMonState::mCOUNT];
    static const std::string macroStateNames[FastMonState::MCOUNT];
    static const std::string inputStateNames[FastMonState::inCOUNT];
    // Reserved names for microstates
    static const std::string nopath_;
    FastMonitoringService(const edm::ParameterSet&, edm::ActivityRegistry&);
    ~FastMonitoringService() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    std::string makePathLegendaJson();
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

    //this is still needed for use in special functions like DQM which are in turn framework services
    void setMicroState(FastMonState::Microstate);
    void setMicroState(edm::StreamID, FastMonState::Microstate);

    void accumulateFileSize(unsigned int lumi, unsigned long fileSize);
    void startedLookingForFile();
    void stoppedLookingForFile(unsigned int lumi);
    void reportLockWait(unsigned int ls, double waitTime, unsigned int lockCount);
    unsigned int getEventsProcessedForLumi(unsigned int lumi, bool* abortFlag = nullptr);
    bool getAbortFlagForLumi(unsigned int lumi);
    bool shouldWriteFiles(unsigned int lumi, unsigned int* proc = nullptr) {
      unsigned int processed = getEventsProcessedForLumi(lumi);
      if (proc)
        *proc = processed;
      return !getAbortFlagForLumi(lumi);
    }
    std::string getRunDirName() const { return runDirectory_.stem().string(); }
    void setInputSource(FedRawDataInputSource* inputSource) { inputSource_ = inputSource; }
    void setInState(FastMonState::InputState inputState) { inputState_ = inputState; }
    void setInStateSup(FastMonState::InputState inputState) { inputSupervisorState_ = inputState; }

  private:
    void doSnapshot(const unsigned int ls, const bool isGlobalEOL);

    void snapshotRunner();

    //the actual monitoring thread is held by a separate class object for ease of maintenance
    std::shared_ptr<FastMonitoringThread> fmt_;
    //Encoding encModule_;
    //std::vector<Encoding> encPath_;
    FedRawDataInputSource* inputSource_ = nullptr;
    std::atomic<FastMonState::InputState> inputState_{FastMonState::InputState::inInit};
    std::atomic<FastMonState::InputState> inputSupervisorState_{FastMonState::InputState::inInit};

    unsigned int nStreams_;
    unsigned int nThreads_;
    int sleepTime_;
    unsigned int fastMonIntervals_;
    unsigned int snapCounter_ = 0;
    std::string microstateDefPath_, fastMicrostateDefPath_;
    std::string fastName_, fastPath_, slowName_;
    bool filePerFwkStream_;

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

    std::vector<std::atomic<bool>*> collectedPathList_;
    std::vector<bool> pathNamesReady_;

    std::filesystem::path workingDirectory_, runDirectory_;

    bool threadIDAvailable_ = false;

    std::atomic<unsigned long> totalEventsProcessed_;

    std::string moduleLegendFile_;
    std::string moduleLegendFileJson_;
    std::string pathLegendFile_;
    std::string pathLegendFileJson_;
    std::string inputLegendFileJson_;
    unsigned int nOutputModules_ = 0;

    std::atomic<bool> monInit_;
    bool exception_detected_ = false;
    std::vector<unsigned int> exceptionInLS_;
    std::vector<std::string> fastPathList_;
  };

}  // namespace evf

#endif
