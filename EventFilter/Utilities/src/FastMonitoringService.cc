#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/FastMonitoringThread.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/FedRawDataInputSource.h"
#include "EventFilter/Utilities/interface/DAQSource.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <iostream>
#include <iomanip>
#include <sys/time.h>

using namespace jsoncollector;

constexpr double throughputFactor() { return (1000000) / double(1024 * 1024); }

namespace evf {

  const edm::ModuleDescription FastMonitoringService::reservedMicroStateNames[FastMonState::mCOUNT] = {
      edm::ModuleDescription("Dummy", "Invalid"),
      edm::ModuleDescription("Dummy", "Idle"),
      edm::ModuleDescription("Dummy", "FwkOvhSrc"),
      edm::ModuleDescription("Dummy", "FwkOvhMod"),  //set post produce, analyze or filter
      edm::ModuleDescription("Dummy", "FwkEoL"),
      edm::ModuleDescription("Dummy", "Input"),
      edm::ModuleDescription("Dummy", "DQM"),
      edm::ModuleDescription("Dummy", "BoL"),
      edm::ModuleDescription("Dummy", "EoL"),
      edm::ModuleDescription("Dummy", "GlobalEoL")};

  const std::string FastMonitoringService::macroStateNames[FastMonState::MCOUNT] = {"Init",
                                                                                    "JobReady",
                                                                                    "RunGiven",
                                                                                    "Running",
                                                                                    "Stopping",
                                                                                    "Done",
                                                                                    "JobEnded",
                                                                                    "Error",
                                                                                    "ErrorEnded",
                                                                                    "End",
                                                                                    "Invalid"};

  const std::string FastMonitoringService::inputStateNames[FastMonState::inCOUNT] = {
      "Ignore",
      "Init",
      "WaitInput",
      "NewLumi",
      "NewLumiBusyEndingLS",
      "NewLumiIdleEndingLS",
      "RunEnd",
      "ProcessingFile",
      "WaitChunk",
      "ChunkReceived",
      "ChecksumEvent",
      "CachedEvent",
      "ReadEvent",
      "ReadCleanup",
      "NoRequest",
      "NoRequestWithIdleThreads",
      "NoRequestWithGlobalEoL",
      "NoRequestWithEoLThreads",
      "SupFileLimit",
      "SupWaitFreeChunk",
      "SupWaitFreeChunkCopying",
      "SupWaitFreeThread",
      "SupWaitFreeThreadCopying",
      "SupBusy",
      "SupLockPolling",
      "SupLockPollingCopying",
      "SupNoFile",
      "SupNewFile",
      "SupNewFileWaitThreadCopying",
      "SupNewFileWaitThread",
      "SupNewFileWaitChunkCopying",
      "SupNewFileWaitChunk",
      "WaitInput_fileLimit",
      "WaitInput_waitFreeChunk",
      "WaitInput_waitFreeChunkCopying",
      "WaitInput_waitFreeThread",
      "WaitInput_waitFreeThreadCopying",
      "WaitInput_busy",
      "WaitInput_lockPolling",
      "WaitInput_lockPollingCopying",
      "WaitInput_runEnd",
      "WaitInput_noFile",
      "WaitInput_newFile",
      "WaitInput_newFileWaitThreadCopying",
      "WaitInput_newFileWaitThread",
      "WaitInput_newFileWaitChunkCopying",
      "WaitInput_newFileWaitChunk",
      "WaitChunk_fileLimit",
      "WaitChunk_waitFreeChunk",
      "WaitChunk_waitFreeChunkCopying",
      "WaitChunk_waitFreeThread",
      "WaitChunk_waitFreeThreadCopying",
      "WaitChunk_busy",
      "WaitChunk_lockPolling",
      "WaitChunk_lockPollingCopying",
      "WaitChunk_runEnd",
      "WaitChunk_noFile",
      "WaitChunk_newFile",
      "WaitChunk_newFileWaitThreadCopying",
      "WaitChunk_newFileWaitThread",
      "WaitChunk_newFileWaitChunkCopying",
      "WaitChunk_newFileWaitChunk",
      "inSupThrottled",
      "inThrottled"};

  const std::string FastMonitoringService::nopath_ = "NoPath";

  FastMonitoringService::FastMonitoringService(const edm::ParameterSet& iPS, edm::ActivityRegistry& reg)
      : MicroStateService(iPS, reg),
        fmt_(new FastMonitoringThread()),
        nStreams_(0)  //until initialized
        ,
        sleepTime_(iPS.getUntrackedParameter<int>("sleepTime", 1)),
        fastMonIntervals_(iPS.getUntrackedParameter<unsigned int>("fastMonIntervals", 2)),
        fastName_("fastmoni"),
        slowName_("slowmoni"),
        filePerFwkStream_(iPS.getUntrackedParameter<bool>("filePerFwkStream", false)),
        totalEventsProcessed_(0),
        verbose_(iPS.getUntrackedParameter<bool>("verbose")) {
    reg.watchPreallocate(this, &FastMonitoringService::preallocate);  //receiving information on number of threads
    reg.watchJobFailure(this, &FastMonitoringService::jobFailure);    //global

    reg.watchPreBeginJob(this, &FastMonitoringService::preBeginJob);
    reg.watchPreModuleBeginJob(this, &FastMonitoringService::preModuleBeginJob);  //global
    reg.watchPostBeginJob(this, &FastMonitoringService::postBeginJob);
    reg.watchPostEndJob(this, &FastMonitoringService::postEndJob);

    reg.watchPreGlobalBeginLumi(this, &FastMonitoringService::preGlobalBeginLumi);  //global lumi
    reg.watchPreGlobalEndLumi(this, &FastMonitoringService::preGlobalEndLumi);
    reg.watchPostGlobalEndLumi(this, &FastMonitoringService::postGlobalEndLumi);

    reg.watchPreStreamBeginLumi(this, &FastMonitoringService::preStreamBeginLumi);  //stream lumi
    reg.watchPostStreamBeginLumi(this, &FastMonitoringService::postStreamBeginLumi);
    reg.watchPreStreamEndLumi(this, &FastMonitoringService::preStreamEndLumi);
    reg.watchPostStreamEndLumi(this, &FastMonitoringService::postStreamEndLumi);

    reg.watchPrePathEvent(this, &FastMonitoringService::prePathEvent);

    reg.watchPreEvent(this, &FastMonitoringService::preEvent);  //stream
    reg.watchPostEvent(this, &FastMonitoringService::postEvent);

    reg.watchPreSourceEvent(this, &FastMonitoringService::preSourceEvent);  //source (with streamID of requestor)
    reg.watchPostSourceEvent(this, &FastMonitoringService::postSourceEvent);

    reg.watchPreModuleEventAcquire(this, &FastMonitoringService::preModuleEventAcquire);  //stream
    reg.watchPostModuleEventAcquire(this, &FastMonitoringService::postModuleEventAcquire);

    reg.watchPreModuleEvent(this, &FastMonitoringService::preModuleEvent);  //stream
    reg.watchPostModuleEvent(this, &FastMonitoringService::postModuleEvent);

    reg.watchPreStreamEarlyTermination(this, &FastMonitoringService::preStreamEarlyTermination);
    reg.watchPreGlobalEarlyTermination(this, &FastMonitoringService::preGlobalEarlyTermination);
    reg.watchPreSourceEarlyTermination(this, &FastMonitoringService::preSourceEarlyTermination);

    //find microstate definition path (required by the module)
    struct stat statbuf;
    std::string microstateBaseSuffix = "src/EventFilter/Utilities/plugins/microstatedef.jsd";
    std::string microstatePath = std::string(std::getenv("CMSSW_BASE")) + "/" + microstateBaseSuffix;
    if (stat(microstatePath.c_str(), &statbuf)) {
      microstatePath = std::string(std::getenv("CMSSW_RELEASE_BASE")) + "/" + microstateBaseSuffix;
      if (stat(microstatePath.c_str(), &statbuf)) {
        microstatePath = microstateBaseSuffix;
        if (stat(microstatePath.c_str(), &statbuf))
          throw cms::Exception("FastMonitoringService") << "microstate definition file not found";
      }
    }
    fastMicrostateDefPath_ = microstateDefPath_ = microstatePath;
  }

  FastMonitoringService::~FastMonitoringService() {}

  void FastMonitoringService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("Service for File-based DAQ monitoring and event accounting");
    desc.addUntracked<int>("sleepTime", 1)->setComment("Sleep time of the monitoring thread");
    desc.addUntracked<unsigned int>("fastMonIntervals", 2)
        ->setComment("Modulo of sleepTime intervals on which fastmon file is written out");
    desc.addUntracked<bool>("filePerFwkStream", false)
        ->setComment("Switches on monitoring output per framework stream");
    desc.addUntracked<bool>("verbose", false)->setComment("Set to use LogInfo messages from the monitoring thread");
    desc.setAllowAnything();
    descriptions.add("FastMonitoringService", desc);
  }

  std::string FastMonitoringService::makePathLegendaJson() {
    Json::Value legendaVector(Json::arrayValue);
    for (int i = 0; i < fmt_->m_data.encPath_[0].current_; i++)
      legendaVector.append(Json::Value(*(static_cast<const std::string*>(fmt_->m_data.encPath_[0].decode(i)))));
    Json::Value valReserved(nReservedPaths);
    Json::Value pathLegend;
    pathLegend["names"] = legendaVector;
    pathLegend["reserved"] = valReserved;
    Json::StyledWriter writer;
    return writer.write(pathLegend);
  }

  std::string FastMonitoringService::makeModuleLegendaJson() {
    Json::Value legendaVector(Json::arrayValue);
    for (int i = 0; i < fmt_->m_data.encModule_.current_; i++)
      legendaVector.append(
          Json::Value((static_cast<const edm::ModuleDescription*>(fmt_->m_data.encModule_.decode(i)))->moduleLabel()));
    //duplicate modules adding a list for acquire states (not all modules actually have it)
    for (int i = 0; i < fmt_->m_data.encModule_.current_; i++)
      legendaVector.append(Json::Value(
          (static_cast<const edm::ModuleDescription*>(fmt_->m_data.encModule_.decode(i)))->moduleLabel() + "__ACQ"));
    Json::Value valReserved(nReservedModules);
    Json::Value valSpecial(nSpecialModules);
    Json::Value valOutputModules(nOutputModules_);
    Json::Value moduleLegend;
    moduleLegend["names"] = legendaVector;
    moduleLegend["reserved"] = valReserved;
    moduleLegend["special"] = valSpecial;
    moduleLegend["output"] = valOutputModules;
    Json::StyledWriter writer;
    return writer.write(moduleLegend);
  }

  std::string FastMonitoringService::makeInputLegendaJson() {
    Json::Value legendaVector(Json::arrayValue);
    for (int i = 0; i < FastMonState::inCOUNT; i++)
      legendaVector.append(Json::Value(inputStateNames[i]));
    Json::Value moduleLegend;
    moduleLegend["names"] = legendaVector;
    Json::StyledWriter writer;
    return writer.write(moduleLegend);
  }

  void FastMonitoringService::preallocate(edm::service::SystemBounds const& bounds) {
    nStreams_ = bounds.maxNumberOfStreams();
    nThreads_ = bounds.maxNumberOfThreads();
    //this should already be >=1
    if (nStreams_ == 0)
      nStreams_ = 1;
    if (nThreads_ == 0)
      nThreads_ = 1;
  }

  void FastMonitoringService::preBeginJob(edm::PathsAndConsumesOfModulesBase const& pathsInfo,
                                          edm::ProcessContext const& pc) {
    // FIND RUN DIRECTORY
    // The run dir should be set via the configuration of EvFDaqDirector

    if (edm::Service<evf::EvFDaqDirector>().operator->() == nullptr) {
      throw cms::Exception("FastMonitoringService") << "EvFDaqDirector is not present";
    }
    std::filesystem::path runDirectory{edm::Service<evf::EvFDaqDirector>()->baseRunDir()};
    workingDirectory_ = runDirectory_ = runDirectory;
    workingDirectory_ /= "mon";

    if (!std::filesystem::is_directory(workingDirectory_)) {
      LogDebug("FastMonitoringService") << "<MON> DIR NOT FOUND! Trying to create -: " << workingDirectory_.string();
      std::filesystem::create_directories(workingDirectory_);
      if (!std::filesystem::is_directory(workingDirectory_))
        edm::LogWarning("FastMonitoringService") << "Unable to create <MON> DIR -: " << workingDirectory_.string()
                                                 << ". No monitoring data will be written.";
    }

    std::ostringstream fastFileName;

    fastFileName << fastName_ << "_pid" << std::setfill('0') << std::setw(5) << getpid() << ".fast";
    std::filesystem::path fast = workingDirectory_;
    fast /= fastFileName.str();
    fastPath_ = fast.string();
    if (filePerFwkStream_)
      for (unsigned int i = 0; i < nStreams_; i++) {
        std::ostringstream fastFileNameTid;
        fastFileNameTid << fastName_ << "_pid" << std::setfill('0') << std::setw(5) << getpid() << "_tid" << i
                        << ".fast";
        std::filesystem::path fastTid = workingDirectory_;
        fastTid /= fastFileNameTid.str();
        fastPathList_.push_back(fastTid.string());
      }

    std::ostringstream moduleLegFile;
    std::ostringstream moduleLegFileJson;
    moduleLegFile << "microstatelegend_pid" << std::setfill('0') << std::setw(5) << getpid() << ".leg";
    moduleLegFileJson << "microstatelegend_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsn";
    moduleLegendFile_ = (workingDirectory_ / moduleLegFile.str()).string();
    moduleLegendFileJson_ = (workingDirectory_ / moduleLegFileJson.str()).string();

    std::ostringstream pathLegFile;
    std::ostringstream pathLegFileJson;
    pathLegFile << "pathlegend_pid" << std::setfill('0') << std::setw(5) << getpid() << ".leg";
    pathLegendFile_ = (workingDirectory_ / pathLegFile.str()).string();
    pathLegFileJson << "pathlegend_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsn";
    pathLegendFileJson_ = (workingDirectory_ / pathLegFileJson.str()).string();

    std::ostringstream inputLegFileJson;
    inputLegFileJson << "inputlegend_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsn";
    inputLegendFileJson_ = (workingDirectory_ / inputLegFileJson.str()).string();

    LogDebug("FastMonitoringService") << "Initializing FastMonitor with microstate def path -: " << microstateDefPath_;
    //<< encPath_.current_ + 1 << " " << encModule_.current_ + 1

    /*
     * initialize the fast monitor with:
     * vector of pointers to monitorable parameters
     * path to definition
     *
     */

    fmt_->m_data.macrostate_ = FastMonState::sInit;

    for (unsigned int i = 0; i < (FastMonState::mCOUNT); i++)
      fmt_->m_data.encModule_.updateReserved(static_cast<const void*>(reservedMicroStateNames + i));
    fmt_->m_data.encModule_.completeReservedWithDummies();

    for (unsigned int i = 0; i < nStreams_; i++) {
      fmt_->m_data.ministate_.emplace_back(&nopath_);
      fmt_->m_data.microstate_.emplace_back(&reservedMicroStateNames[FastMonState::mInvalid]);
      fmt_->m_data.microstateAcqFlag_.push_back(0);

      //for synchronization
      streamCounterUpdating_.push_back(new std::atomic<bool>(false));

      //path (mini) state
      fmt_->m_data.encPath_.emplace_back(0);
      fmt_->m_data.encPath_[i].update(static_cast<const void*>(&nopath_));

      for (auto& path : pathsInfo.paths()) {
        fmt_->m_data.encPath_[i].updatePreinit(path);
      }
      for (auto& endPath : pathsInfo.endPaths()) {
        fmt_->m_data.encPath_[i].updatePreinit(endPath);
      }
    }
    //for (unsigned int i=0;i<nThreads_;i++)
    //  threadMicrostate_.push_back(&reservedMicroStateNames[mInvalid]);

    //initial size until we detect number of bins
    fmt_->m_data.macrostateBins_ = FastMonState::MCOUNT;
    fmt_->m_data.microstateBins_ = 0;
    fmt_->m_data.inputstateBins_ = FastMonState::inCOUNT;
    fmt_->m_data.ministateBins_ = fmt_->m_data.encPath_[0].vecsize();

    lastGlobalLumi_ = 0;
    isInitTransition_ = true;
    lumiFromSource_ = 0;

    //startup monitoring
    fmt_->resetFastMonitor(microstateDefPath_, fastMicrostateDefPath_);
    fmt_->jsonMonitor_->setNStreams(nStreams_);
    fmt_->m_data.registerVariables(fmt_->jsonMonitor_.get(), nStreams_, threadIDAvailable_ ? nThreads_ : 0);
    monInit_.store(false, std::memory_order_release);
    if (sleepTime_ > 0)
      fmt_->start(&FastMonitoringService::snapshotRunner, this);

    //this definition needs: #include "tbb/compat/thread"
    //however this would results in TBB imeplementation replacing std::thread
    //(both supposedly call pthread_self())
    //number of threads created in process could be obtained from /proc,
    //assuming that all posix threads are true kernel threads capable of running in parallel

    //#if TBB_IMPLEMENT_CPP0X
    ////std::cout << "TBB thread id:" <<  tbb::thread::id() << std::endl;
    //threadIDAvailable_=true;
    //#endif
  }

  void FastMonitoringService::preStreamEarlyTermination(edm::StreamContext const& sc, edm::TerminationOrigin to) {
    std::string context;
    if (to == edm::TerminationOrigin::ExceptionFromThisContext)
      context = " FromThisContext ";
    if (to == edm::TerminationOrigin::ExceptionFromAnotherContext)
      context = " FromAnotherContext";
    if (to == edm::TerminationOrigin::ExternalSignal)
      context = " FromExternalSignal";
    edm::LogWarning("FastMonitoringService")
        << " STREAM " << sc.streamID().value() << " earlyTermination -: ID:" << sc.eventID()
        << " LS:" << sc.eventID().luminosityBlock() << " " << context;
    std::lock_guard<std::mutex> lock(fmt_->monlock_);
    exceptionInLS_.push_back(sc.eventID().luminosityBlock());
    has_data_exception_.store(true);
  }

  void FastMonitoringService::preGlobalEarlyTermination(edm::GlobalContext const& gc, edm::TerminationOrigin to) {
    std::string context;
    if (to == edm::TerminationOrigin::ExceptionFromThisContext)
      context = " FromThisContext ";
    if (to == edm::TerminationOrigin::ExceptionFromAnotherContext)
      context = " FromAnotherContext";
    if (to == edm::TerminationOrigin::ExternalSignal)
      context = " FromExternalSignal";
    edm::LogWarning("FastMonitoringService")
        << " GLOBAL "
        << "earlyTermination -: LS:" << gc.luminosityBlockID().luminosityBlock() << " " << context;
    std::lock_guard<std::mutex> lock(fmt_->monlock_);
    exceptionInLS_.push_back(gc.luminosityBlockID().luminosityBlock());
    has_data_exception_.store(true);
  }

  void FastMonitoringService::preSourceEarlyTermination(edm::TerminationOrigin to) {
    std::string context;
    if (to == edm::TerminationOrigin::ExceptionFromThisContext)
      context = " FromThisContext ";
    if (to == edm::TerminationOrigin::ExceptionFromAnotherContext)
      context = " FromAnotherContext";
    if (to == edm::TerminationOrigin::ExternalSignal)
      context = " FromExternalSignal";
    edm::LogWarning("FastMonitoringService") << " SOURCE "
                                             << "earlyTermination -: " << context;
    std::lock_guard<std::mutex> lock(fmt_->monlock_);
    exception_detected_ = true;
    has_source_exception_.store(true);
    has_data_exception_.store(true);
  }

  void FastMonitoringService::setExceptionDetected(unsigned int ls) {
    std::lock_guard<std::mutex> lock(fmt_->monlock_);
    if (!ls)
      exception_detected_ = true;
    else
      exceptionInLS_.push_back(ls);
  }

  bool FastMonitoringService::exceptionDetected() const {
    return has_source_exception_.load() || has_data_exception_.load();
  }

  bool FastMonitoringService::isExceptionOnData(unsigned int ls) {
    if (!has_data_exception_.load())
      return false;
    if (has_source_exception_.load())
      return true;
    std::lock_guard<std::mutex> lock(fmt_->monlock_);
    for (auto ex : exceptionInLS_) {
      if (ls == ex)
        return true;
    }
    return false;
  }

  void FastMonitoringService::jobFailure() { fmt_->m_data.macrostate_ = FastMonState::sError; }

  //new output module name is stream
  void FastMonitoringService::preModuleBeginJob(const edm::ModuleDescription& desc) {
    std::lock_guard<std::mutex> lock(fmt_->monlock_);
    //std::cout << " Pre module Begin Job module: " << desc.moduleName() << std::endl;

    //build a map of modules keyed by their module description address
    //here we need to treat output modules in a special way so they can be easily singled out
    if (desc.moduleName() == "Stream" || desc.moduleName() == "GlobalEvFOutputModule" ||
        desc.moduleName() == "EvFOutputModule" || desc.moduleName() == "EventStreamFileWriter" ||
        desc.moduleName() == "PoolOutputModule") {
      fmt_->m_data.encModule_.updateReserved((void*)&desc);
      nOutputModules_++;
    } else
      fmt_->m_data.encModule_.update((void*)&desc);
  }

  void FastMonitoringService::postBeginJob() {
    std::string&& moduleLegStrJson = makeModuleLegendaJson();
    FileIO::writeStringToFile(moduleLegendFileJson_, moduleLegStrJson);

    std::string inputLegendStrJson = makeInputLegendaJson();
    FileIO::writeStringToFile(inputLegendFileJson_, inputLegendStrJson);

    std::string pathLegendStrJson = makePathLegendaJson();
    FileIO::writeStringToFile(pathLegendFileJson_, pathLegendStrJson);

    fmt_->m_data.macrostate_ = FastMonState::sJobReady;

    //update number of entries in module histogram
    std::lock_guard<std::mutex> lock(fmt_->monlock_);
    //double the size to add post-acquire states
    fmt_->m_data.microstateBins_ = fmt_->m_data.encModule_.vecsize() * 2;
  }

  void FastMonitoringService::postEndJob() {
    fmt_->m_data.macrostate_ = FastMonState::sJobEnded;
    fmt_->stop();
  }

  void FastMonitoringService::postGlobalBeginRun(edm::GlobalContext const& gc) {
    fmt_->m_data.macrostate_ = FastMonState::sRunning;
    isInitTransition_ = false;
  }

  void FastMonitoringService::preGlobalBeginLumi(edm::GlobalContext const& gc) {
    timeval lumiStartTime;
    gettimeofday(&lumiStartTime, nullptr);
    unsigned int newLumi = gc.luminosityBlockID().luminosityBlock();
    lastGlobalLumi_ = newLumi;

    std::lock_guard<std::mutex> lock(fmt_->monlock_);
    lumiStartTime_[newLumi] = lumiStartTime;
  }

  void FastMonitoringService::preGlobalEndLumi(edm::GlobalContext const& gc) {
    unsigned int lumi = gc.luminosityBlockID().luminosityBlock();
    LogDebug("FastMonitoringService") << "Lumi ended. Writing JSON information. LUMI -: " << lumi;
    timeval lumiStopTime;
    gettimeofday(&lumiStopTime, nullptr);

    std::lock_guard<std::mutex> lock(fmt_->monlock_);

    // Compute throughput
    timeval stt = lumiStartTime_[lumi];
    lumiStartTime_.erase(lumi);
    unsigned long usecondsForLumi = (lumiStopTime.tv_sec - stt.tv_sec) * 1000000 + (lumiStopTime.tv_usec - stt.tv_usec);
    unsigned long accuSize = accuSize_.find(lumi) == accuSize_.end() ? 0 : accuSize_[lumi];
    accuSize_.erase(lumi);
    double throughput = throughputFactor() * double(accuSize) / double(usecondsForLumi);
    //store to registered variable
    fmt_->m_data.fastThroughputJ_.value() = throughput;

    //update
    doSnapshot(lumi, true);

    //retrieve one result we need (todo: sanity check if it's found)
    IntJ* lumiProcessedJptr = dynamic_cast<IntJ*>(fmt_->jsonMonitor_->getMergedIntJForLumi("Processed", lumi));
    if (!lumiProcessedJptr)
      throw cms::Exception("FastMonitoringService") << "Internal error: got null pointer from FastMonitor";
    processedEventsPerLumi_[lumi] = std::pair<unsigned int, bool>(lumiProcessedJptr->value(), false);

    //checking if exception has been thrown (in case of Global/Stream early termination, for this LS)
    bool exception_detected = exception_detected_;
    for (auto ex : exceptionInLS_)
      if (lumi == ex)
        exception_detected = true;

    if (edm::shutdown_flag || exception_detected) {
      edm::LogInfo("FastMonitoringService")
          << "Run interrupted. Skip writing EoL information -: " << processedEventsPerLumi_[lumi].first
          << " events were processed in LUMI " << lumi;
      //this will prevent output modules from producing json file for possibly incomplete lumi
      processedEventsPerLumi_[lumi].first = 0;
      processedEventsPerLumi_[lumi].second = true;
      //disable this exception, so service can be used standalone (will be thrown if output module asks for this information)
      //throw cms::Exception("FastMonitoringService") << "SOURCE did not send update for lumi block. LUMI -:" << lumi;
      return;
    }

    if (inputSource_ || daqInputSource_) {
      auto sourceReport =
          inputSource_ ? inputSource_->getEventReport(lumi, true) : daqInputSource_->getEventReport(lumi, true);
      if (sourceReport.first) {
        if (sourceReport.second != processedEventsPerLumi_[lumi].first) {
          throw cms::Exception("FastMonitoringService") << "MISMATCH with SOURCE update. LUMI -: " << lumi
                                                        << ", events(processed):" << processedEventsPerLumi_[lumi].first
                                                        << " events(source):" << sourceReport.second;
        }
      }
    }

    edm::LogInfo("FastMonitoringService")
        << "Statistics for lumisection -: lumi = " << lumi << " events = " << lumiProcessedJptr->value()
        << " time = " << usecondsForLumi / 1000000 << " size = " << accuSize << " thr = " << throughput;
    delete lumiProcessedJptr;

    //full global and stream merge&output for this lumi

    // create file name for slow monitoring file
    bool output = sleepTime_ > 0;
    if (filePerFwkStream_) {
      std::stringstream slowFileNameStem;
      slowFileNameStem << slowName_ << "_ls" << std::setfill('0') << std::setw(4) << lumi << "_pid" << std::setfill('0')
                       << std::setw(5) << getpid();
      std::filesystem::path slow = workingDirectory_;
      slow /= slowFileNameStem.str();
      fmt_->jsonMonitor_->outputFullJSONs(slow.string(), ".jsn", lumi, output);
    } else {
      std::stringstream slowFileName;
      slowFileName << slowName_ << "_ls" << std::setfill('0') << std::setw(4) << lumi << "_pid" << std::setfill('0')
                   << std::setw(5) << getpid() << ".jsn";
      std::filesystem::path slow = workingDirectory_;
      slow /= slowFileName.str();
      //full global and stream merge and JSON write for this lumi
      fmt_->jsonMonitor_->outputFullJSON(slow.string(), lumi, output);
    }
    fmt_->jsonMonitor_->discardCollected(lumi);  //we don't do further updates for this lumi
  }

  void FastMonitoringService::postGlobalEndLumi(edm::GlobalContext const& gc) {
    std::lock_guard<std::mutex> lock(fmt_->monlock_);
    unsigned int lumi = gc.luminosityBlockID().luminosityBlock();
    //LS monitoring snapshot with input source data has been taken in previous callback
    avgLeadTime_.erase(lumi);
    filesProcessedDuringLumi_.erase(lumi);
    lockStatsDuringLumi_.erase(lumi);

    //output module already used this in end lumi (this could be migrated to EvFDaqDirector as it is essential for FFF bookkeeping)
    processedEventsPerLumi_.erase(lumi);
  }

  void FastMonitoringService::preStreamBeginLumi(edm::StreamContext const& sc) {
    unsigned int sid = sc.streamID().value();

    std::lock_guard<std::mutex> lock(fmt_->monlock_);
    fmt_->m_data.streamLumi_[sid] = sc.eventID().luminosityBlock();

    //reset collected values for this stream
    *(fmt_->m_data.processed_[sid]) = 0;

    fmt_->m_data.ministate_[sid] = &nopath_;
    fmt_->m_data.microstate_[sid] = &reservedMicroStateNames[FastMonState::mBoL];
  }

  void FastMonitoringService::postStreamBeginLumi(edm::StreamContext const& sc) {
    fmt_->m_data.microstate_[sc.streamID().value()] = &reservedMicroStateNames[FastMonState::mIdle];
  }

  void FastMonitoringService::preStreamEndLumi(edm::StreamContext const& sc) {
    unsigned int sid = sc.streamID().value();
    std::lock_guard<std::mutex> lock(fmt_->monlock_);

    //update processed count to be complete at this time
    //doStreamEOLSnapshot(sc.eventID().luminosityBlock(), sid);
    fmt_->jsonMonitor_->snapStreamAtomic(sc.eventID().luminosityBlock(), sid);
    //reset this in case stream does not get notified of next lumi (we keep processed events only)
    fmt_->m_data.ministate_[sid] = &nopath_;
    fmt_->m_data.microstate_[sid] = &reservedMicroStateNames[FastMonState::mEoL];
  }
  void FastMonitoringService::postStreamEndLumi(edm::StreamContext const& sc) {
    fmt_->m_data.microstate_[sc.streamID().value()] = &reservedMicroStateNames[FastMonState::mFwkEoL];
  }

  void FastMonitoringService::prePathEvent(edm::StreamContext const& sc, edm::PathContext const& pc) {
    fmt_->m_data.ministate_[sc.streamID()] = &(pc.pathName());
  }

  void FastMonitoringService::preEvent(edm::StreamContext const& sc) {}

  void FastMonitoringService::postEvent(edm::StreamContext const& sc) {
    fmt_->m_data.microstate_[sc.streamID()] = &reservedMicroStateNames[FastMonState::mIdle];

    fmt_->m_data.ministate_[sc.streamID()] = &nopath_;

    (*(fmt_->m_data.processed_[sc.streamID()]))++;

    //fast path counter (events accumulated in a run)
    unsigned long res = totalEventsProcessed_.fetch_add(1, std::memory_order_relaxed);
    fmt_->m_data.fastPathProcessedJ_ = res + 1;
  }

  void FastMonitoringService::preSourceEvent(edm::StreamID sid) {
    fmt_->m_data.microstate_[sid.value()] = &reservedMicroStateNames[FastMonState::mInput];
  }

  void FastMonitoringService::postSourceEvent(edm::StreamID sid) {
    fmt_->m_data.microstate_[sid.value()] = &reservedMicroStateNames[FastMonState::mFwkOvhSrc];
  }

  void FastMonitoringService::preModuleEventAcquire(edm::StreamContext const& sc,
                                                    edm::ModuleCallingContext const& mcc) {
    fmt_->m_data.microstate_[sc.streamID().value()] = (void*)(mcc.moduleDescription());
  }

  void FastMonitoringService::postModuleEventAcquire(edm::StreamContext const& sc,
                                                     edm::ModuleCallingContext const& mcc) {
    //fmt_->m_data.microstate_[sc.streamID().value()] = (void*)(mcc.moduleDescription());
    fmt_->m_data.microstateAcqFlag_[sc.streamID().value()] = 1;
  }

  void FastMonitoringService::preModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
    fmt_->m_data.microstate_[sc.streamID().value()] = (void*)(mcc.moduleDescription());
    fmt_->m_data.microstateAcqFlag_[sc.streamID().value()] = 0;
  }

  void FastMonitoringService::postModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
    fmt_->m_data.microstate_[sc.streamID().value()] = &reservedMicroStateNames[FastMonState::mFwkOvhMod];
  }

  //FUNCTIONS CALLED FROM OUTSIDE

  //this is for old-fashioned service that is not thread safe and can block other streams
  //(we assume the worst case - everything is blocked)
  void FastMonitoringService::setMicroState(FastMonState::Microstate m) {
    for (unsigned int i = 0; i < nStreams_; i++)
      fmt_->m_data.microstate_[i] = &reservedMicroStateNames[m];
  }

  //this is for services that are multithreading-enabled or rarely blocks other streams
  void FastMonitoringService::setMicroState(edm::StreamID sid, FastMonState::Microstate m) {
    fmt_->m_data.microstate_[sid] = &reservedMicroStateNames[m];
  }

  //from source
  void FastMonitoringService::accumulateFileSize(unsigned int lumi, unsigned long fileSize) {
    std::lock_guard<std::mutex> lock(fmt_->monlock_);

    if (accuSize_.find(lumi) == accuSize_.end())
      accuSize_[lumi] = fileSize;
    else
      accuSize_[lumi] += fileSize;

    if (filesProcessedDuringLumi_.find(lumi) == filesProcessedDuringLumi_.end())
      filesProcessedDuringLumi_[lumi] = 1;
    else
      filesProcessedDuringLumi_[lumi]++;
  }

  void FastMonitoringService::startedLookingForFile() {
    gettimeofday(&fileLookStart_, nullptr);
    /*
	 std::cout << "Started looking for .raw file at: s=" << fileLookStart_.tv_sec << ": ms = "
	 << fileLookStart_.tv_usec / 1000.0 << std::endl;
	 */
  }

  void FastMonitoringService::stoppedLookingForFile(unsigned int lumi) {
    gettimeofday(&fileLookStop_, nullptr);
    /*
	 std::cout << "Stopped looking for .raw file at: s=" << fileLookStop_.tv_sec << ": ms = "
	 << fileLookStop_.tv_usec / 1000.0 << std::endl;
	 */
    std::lock_guard<std::mutex> lock(fmt_->monlock_);

    if (lumi > lumiFromSource_) {
      lumiFromSource_ = lumi;
      leadTimes_.clear();
    }
    unsigned long elapsedTime = (fileLookStop_.tv_sec - fileLookStart_.tv_sec) * 1000000  // sec to us
                                + (fileLookStop_.tv_usec - fileLookStart_.tv_usec);       // us
    // add this to lead times for this lumi
    leadTimes_.push_back((double)elapsedTime);

    // recompute average lead time for this lumi
    if (leadTimes_.size() == 1)
      avgLeadTime_[lumi] = leadTimes_[0];
    else {
      double totTime = 0;
      for (unsigned int i = 0; i < leadTimes_.size(); i++)
        totTime += leadTimes_[i];
      avgLeadTime_[lumi] = 0.001 * (totTime / leadTimes_.size());
    }
  }

  void FastMonitoringService::reportLockWait(unsigned int ls, double waitTime, unsigned int lockCount) {
    std::lock_guard<std::mutex> lock(fmt_->monlock_);
    lockStatsDuringLumi_[ls] = std::pair<double, unsigned int>(waitTime, lockCount);
  }

  //for the output module
  unsigned int FastMonitoringService::getEventsProcessedForLumi(unsigned int lumi, bool* abortFlag) {
    std::lock_guard<std::mutex> lock(fmt_->monlock_);

    auto it = processedEventsPerLumi_.find(lumi);
    if (it != processedEventsPerLumi_.end()) {
      unsigned int proc = it->second.first;
      if (abortFlag)
        *abortFlag = it->second.second;
      return proc;
    } else {
      throw cms::Exception("FastMonitoringService")
          << "output module wants already deleted (or never reported by SOURCE) lumisection event count for LUMI -: "
          << lumi;
      return 0;
    }
  }

  //for the output module
  bool FastMonitoringService::getAbortFlagForLumi(unsigned int lumi) {
    std::lock_guard<std::mutex> lock(fmt_->monlock_);

    auto it = processedEventsPerLumi_.find(lumi);
    if (it != processedEventsPerLumi_.end()) {
      unsigned int abortFlag = it->second.second;
      return abortFlag;
    } else {
      throw cms::Exception("FastMonitoringService")
          << "output module wants already deleted (or never reported by SOURCE) lumisection status for LUMI -: "
          << lumi;
      return false;
    }
  }

  // the function to be called in the thread. Thread completes when function returns.
  void FastMonitoringService::snapshotRunner() {
    monInit_.exchange(true, std::memory_order_acquire);
    while (!fmt_->m_stoprequest) {
      std::vector<std::vector<unsigned int>> lastEnc;
      {
        std::unique_lock<std::mutex> lock(fmt_->monlock_);

        doSnapshot(lastGlobalLumi_, false);

        lastEnc.emplace_back(fmt_->m_data.ministateEncoded_);
        lastEnc.emplace_back(fmt_->m_data.microstateEncoded_);

        if (fastMonIntervals_ && (snapCounter_ % fastMonIntervals_) == 0) {
          if (filePerFwkStream_) {
            std::vector<std::string> CSVv;
            for (unsigned int i = 0; i < nStreams_; i++) {
              CSVv.push_back(fmt_->jsonMonitor_->getCSVString((int)i));
            }
            // release mutex before writing out fast path file
            lock.release()->unlock();
            for (unsigned int i = 0; i < nStreams_; i++) {
              if (!CSVv[i].empty())
                fmt_->jsonMonitor_->outputCSV(fastPathList_[i], CSVv[i]);
            }
          } else {
            std::string CSV = fmt_->jsonMonitor_->getCSVString();
            // release mutex before writing out fast path file
            lock.release()->unlock();
            if (!CSV.empty())
              fmt_->jsonMonitor_->outputCSV(fastPath_, CSV);
          }
        }
        snapCounter_++;
      }

      if (verbose_) {
        edm::LogInfo msg("FastMonitoringService");
        auto f = [&](std::vector<unsigned int> const& p) {
          for (unsigned int i = 0; i < nStreams_; i++) {
            if (i == 0)
              msg << "[" << p[i] << ",";
            else if (i <= nStreams_ - 1)
              msg << p[i] << ",";
            else
              msg << p[i] << "]";
          }
        };

        msg << "Current states: Ms=" << fmt_->m_data.fastMacrostateJ_.value() << " ms=";
        f(lastEnc[0]);
        msg << " us=";
        f(lastEnc[1]);
        msg << " is=" << inputStateNames[inputState_] << " iss=" << inputStateNames[inputSupervisorState_];
      }

      ::sleep(sleepTime_);
    }
  }

  void FastMonitoringService::doSnapshot(const unsigned int ls, const bool isGlobalEOL) {
    // update macrostate
    fmt_->m_data.fastMacrostateJ_ = fmt_->m_data.macrostate_;

    std::vector<const void*> microstateCopy(fmt_->m_data.microstate_.begin(), fmt_->m_data.microstate_.end());
    std::vector<unsigned char> microstateAcqCopy(fmt_->m_data.microstateAcqFlag_.begin(),
                                                 fmt_->m_data.microstateAcqFlag_.end());

    if (!isInitTransition_) {
      auto itd = avgLeadTime_.find(ls);
      if (itd != avgLeadTime_.end())
        fmt_->m_data.fastAvgLeadTimeJ_ = itd->second;
      else
        fmt_->m_data.fastAvgLeadTimeJ_ = 0.;

      auto iti = filesProcessedDuringLumi_.find(ls);
      if (iti != filesProcessedDuringLumi_.end())
        fmt_->m_data.fastFilesProcessedJ_ = iti->second;
      else
        fmt_->m_data.fastFilesProcessedJ_ = 0;

      auto itrd = lockStatsDuringLumi_.find(ls);
      if (itrd != lockStatsDuringLumi_.end()) {
        fmt_->m_data.fastLockWaitJ_ = itrd->second.first;
        fmt_->m_data.fastLockCountJ_ = itrd->second.second;
      } else {
        fmt_->m_data.fastLockWaitJ_ = 0.;
        fmt_->m_data.fastLockCountJ_ = 0.;
      }
    }

    for (unsigned int i = 0; i < nStreams_; i++) {
      fmt_->m_data.ministateEncoded_[i] = fmt_->m_data.encPath_[i].encodeString(fmt_->m_data.ministate_[i]);
      if (microstateAcqCopy[i])
        fmt_->m_data.microstateEncoded_[i] =
            fmt_->m_data.microstateBins_ + fmt_->m_data.encModule_.encode(microstateCopy[i]);
      else
        fmt_->m_data.microstateEncoded_[i] = fmt_->m_data.encModule_.encode(microstateCopy[i]);
    }

    bool inputStatePerThread = false;

    if (inputState_ == FastMonState::inWaitInput) {
      switch (inputSupervisorState_) {
        case FastMonState::inSupFileLimit:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_fileLimit;
          break;
        case FastMonState::inSupWaitFreeChunk:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_waitFreeChunk;
          break;
        case FastMonState::inSupWaitFreeChunkCopying:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_waitFreeChunkCopying;
          break;
        case FastMonState::inSupWaitFreeThread:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_waitFreeThread;
          break;
        case FastMonState::inSupWaitFreeThreadCopying:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_waitFreeThreadCopying;
          break;
        case FastMonState::inSupBusy:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_busy;
          break;
        case FastMonState::inSupLockPolling:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_lockPolling;
          break;
        case FastMonState::inSupLockPollingCopying:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_lockPollingCopying;
          break;
        case FastMonState::inRunEnd:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_runEnd;
          break;
        case FastMonState::inSupNoFile:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_noFile;
          break;
        case FastMonState::inSupNewFile:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_newFile;
          break;
        case FastMonState::inSupNewFileWaitThreadCopying:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_newFileWaitThreadCopying;
          break;
        case FastMonState::inSupNewFileWaitThread:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_newFileWaitThread;
          break;
        case FastMonState::inSupNewFileWaitChunkCopying:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_newFileWaitChunkCopying;
          break;
        case FastMonState::inSupNewFileWaitChunk:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput_newFileWaitChunk;
          break;
        default:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitInput;
      }
    } else if (inputState_ == FastMonState::inWaitChunk) {
      switch (inputSupervisorState_) {
        case FastMonState::inSupFileLimit:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_fileLimit;
          break;
        case FastMonState::inSupWaitFreeChunk:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_waitFreeChunk;
          break;
        case FastMonState::inSupWaitFreeChunkCopying:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_waitFreeChunkCopying;
          break;
        case FastMonState::inSupWaitFreeThread:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_waitFreeThread;
          break;
        case FastMonState::inSupWaitFreeThreadCopying:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_waitFreeThreadCopying;
          break;
        case FastMonState::inSupBusy:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_busy;
          break;
        case FastMonState::inSupLockPolling:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_lockPolling;
          break;
        case FastMonState::inSupLockPollingCopying:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_lockPollingCopying;
          break;
        case FastMonState::inRunEnd:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_runEnd;
          break;
        case FastMonState::inSupNoFile:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_noFile;
          break;
        case FastMonState::inSupNewFile:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_newFile;
          break;
        case FastMonState::inSupNewFileWaitThreadCopying:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_newFileWaitThreadCopying;
          break;
        case FastMonState::inSupNewFileWaitThread:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_newFileWaitThread;
          break;
        case FastMonState::inSupNewFileWaitChunkCopying:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_newFileWaitChunkCopying;
          break;
        case FastMonState::inSupNewFileWaitChunk:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk_newFileWaitChunk;
          break;
        default:
          fmt_->m_data.inputState_[0] = FastMonState::inWaitChunk;
      }
    } else if (inputState_ == FastMonState::inNoRequest) {
      inputStatePerThread = true;
      for (unsigned int i = 0; i < nStreams_; i++) {
        if (microstateCopy[i] == &reservedMicroStateNames[FastMonState::mIdle])
          fmt_->m_data.inputState_[i] = FastMonState::inNoRequestWithIdleThreads;
        else if (microstateCopy[i] == &reservedMicroStateNames[FastMonState::mEoL] ||
                 microstateCopy[i] == &reservedMicroStateNames[FastMonState::mFwkEoL])
          fmt_->m_data.inputState_[i] = FastMonState::inNoRequestWithEoLThreads;
        else
          fmt_->m_data.inputState_[i] = FastMonState::inNoRequest;
      }
    } else if (inputState_ == FastMonState::inNewLumi) {
      inputStatePerThread = true;
      for (unsigned int i = 0; i < nStreams_; i++) {
        if (microstateCopy[i] == &reservedMicroStateNames[FastMonState::mEoL] ||
            microstateCopy[i] == &reservedMicroStateNames[FastMonState::mFwkEoL])
          fmt_->m_data.inputState_[i] = FastMonState::inNewLumi;
      }
    } else if (inputSupervisorState_ == FastMonState::inSupThrottled) {
      //apply directly throttled state from supervisor
      fmt_->m_data.inputState_[0] = inputSupervisorState_;
    } else
      fmt_->m_data.inputState_[0] = inputState_;

    //this is same for all streams
    if (!inputStatePerThread)
      for (unsigned int i = 1; i < nStreams_; i++)
        fmt_->m_data.inputState_[i] = fmt_->m_data.inputState_[0];

    if (isGlobalEOL) {  //only update global variables
      fmt_->jsonMonitor_->snapGlobal(ls);
    } else
      fmt_->jsonMonitor_->snap(ls);
  }

  //compatibility
  MicroStateService::MicroStateService(const edm::ParameterSet& iPS, edm::ActivityRegistry& reg) {}

  MicroStateService::~MicroStateService() {}

}  //end namespace evf
