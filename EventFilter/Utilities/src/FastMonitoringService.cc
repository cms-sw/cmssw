#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include <iostream>

#include "FWCore/Framework/interface/Event.h"
#include <iomanip>
#include <sys/time.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/FedRawDataInputSource.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
using namespace jsoncollector;

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

constexpr double throughputFactor() { return (1000000) / double(1024 * 1024); }

static const int nReservedModules = 64;
static const int nSpecialModules = 10;
static const int nReservedPaths = 1;

namespace evf {

  const std::string FastMonitoringService::macroStateNames[FastMonitoringThread::MCOUNT] = {"Init",
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

  const std::string FastMonitoringService::inputStateNames[FastMonitoringThread::inCOUNT] = {
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
      "WaitChunk_newFileWaitChunk"};

  const std::string FastMonitoringService::nopath_ = "NoPath";

  FastMonitoringService::FastMonitoringService(const edm::ParameterSet& iPS, edm::ActivityRegistry& reg)
      : MicroStateService(iPS, reg),
        encModule_(nReservedModules),
        nStreams_(0)  //until initialized
        ,
        sleepTime_(iPS.getUntrackedParameter<int>("sleepTime", 1)),
        fastMonIntervals_(iPS.getUntrackedParameter<unsigned int>("fastMonIntervals", 2)),
        fastName_("fastmoni"),
        slowName_("slowmoni"),
        filePerFwkStream_(iPS.getUntrackedParameter<bool>("filePerFwkStream", false)),
        totalEventsProcessed_(0) {
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

    reg.watchPreModuleEvent(this, &FastMonitoringService::preModuleEvent);    //should be stream
    reg.watchPostModuleEvent(this, &FastMonitoringService::postModuleEvent);  //

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
    desc.setAllowAnything();
    descriptions.add("FastMonitoringService", desc);
  }

  std::string FastMonitoringService::makePathLegendaJson() {
    Json::Value legendaVector(Json::arrayValue);
    for (int i = 0; i < encPath_[0].current_; i++)
      legendaVector.append(Json::Value(*(static_cast<const std::string*>(encPath_[0].decode(i)))));
    Json::Value valReserved(nReservedPaths);
    Json::Value pathLegend;
    pathLegend["names"] = legendaVector;
    pathLegend["reserved"] = valReserved;
    Json::StyledWriter writer;
    return writer.write(pathLegend);
  }

  std::string FastMonitoringService::makeModuleLegendaJson() {
    Json::Value legendaVector(Json::arrayValue);
    for (int i = 0; i < encModule_.current_; i++)
      legendaVector.append(
          Json::Value((static_cast<const edm::ModuleDescription*>(encModule_.decode(i)))->moduleLabel()));
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
    for (int i = 0; i < FastMonitoringThread::inCOUNT; i++)
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

  void FastMonitoringService::preBeginJob(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const& pc) {
    // FIND RUN DIRECTORY
    // The run dir should be set via the configuration of EvFDaqDirector

    if (edm::Service<evf::EvFDaqDirector>().operator->() == nullptr) {
      throw cms::Exception("FastMonitoringService") << "EvFDaqDirector is not present";
    }
    boost::filesystem::path runDirectory{edm::Service<evf::EvFDaqDirector>()->baseRunDir()};
    workingDirectory_ = runDirectory_ = runDirectory;
    workingDirectory_ /= "mon";

    if (!boost::filesystem::is_directory(workingDirectory_)) {
      LogDebug("FastMonitoringService") << "<MON> DIR NOT FOUND! Trying to create -: " << workingDirectory_.string();
      boost::filesystem::create_directories(workingDirectory_);
      if (!boost::filesystem::is_directory(workingDirectory_))
        edm::LogWarning("FastMonitoringService") << "Unable to create <MON> DIR -: " << workingDirectory_.string()
                                                 << ". No monitoring data will be written.";
    }

    std::ostringstream fastFileName;

    fastFileName << fastName_ << "_pid" << std::setfill('0') << std::setw(5) << getpid() << ".fast";
    boost::filesystem::path fast = workingDirectory_;
    fast /= fastFileName.str();
    fastPath_ = fast.string();
    if (filePerFwkStream_)
      for (unsigned int i = 0; i < nStreams_; i++) {
        std::ostringstream fastFileNameTid;
        fastFileNameTid << fastName_ << "_pid" << std::setfill('0') << std::setw(5) << getpid() << "_tid" << i
                        << ".fast";
        boost::filesystem::path fastTid = workingDirectory_;
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

    macrostate_ = FastMonitoringThread::sInit;

    for (unsigned int i = 0; i < (mCOUNT); i++)
      encModule_.updateReserved(static_cast<const void*>(reservedMicroStateNames + i));
    encModule_.completeReservedWithDummies();

    for (unsigned int i = 0; i < nStreams_; i++) {
      ministate_.emplace_back(&nopath_);
      microstate_.emplace_back(&reservedMicroStateNames[mInvalid]);

      //for synchronization
      streamCounterUpdating_.push_back(new std::atomic<bool>(false));

      //path (mini) state
      encPath_.emplace_back(0);
      encPath_[i].update(static_cast<const void*>(&nopath_));
      eventCountForPathInit_.push_back(0);
      firstEventId_.push_back(0);
      collectedPathList_.push_back(new std::atomic<bool>(false));
    }
    //for (unsigned int i=0;i<nThreads_;i++)
    //  threadMicrostate_.push_back(&reservedMicroStateNames[mInvalid]);

    //initial size until we detect number of bins
    fmt_.m_data.macrostateBins_ = FastMonitoringThread::MCOUNT;
    fmt_.m_data.ministateBins_ = 0;
    fmt_.m_data.microstateBins_ = 0;
    fmt_.m_data.inputstateBins_ = FastMonitoringThread::inCOUNT;

    lastGlobalLumi_ = 0;
    isInitTransition_ = true;
    lumiFromSource_ = 0;

    //startup monitoring
    fmt_.resetFastMonitor(microstateDefPath_, fastMicrostateDefPath_);
    fmt_.jsonMonitor_->setNStreams(nStreams_);
    fmt_.m_data.registerVariables(fmt_.jsonMonitor_.get(), nStreams_, threadIDAvailable_ ? nThreads_ : 0);
    monInit_.store(false, std::memory_order_release);
    fmt_.start(&FastMonitoringService::dowork, this);

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
    std::lock_guard<std::mutex> lock(fmt_.monlock_);
    exceptionInLS_.push_back(sc.eventID().luminosityBlock());
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
    std::lock_guard<std::mutex> lock(fmt_.monlock_);
    exceptionInLS_.push_back(gc.luminosityBlockID().luminosityBlock());
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
    std::lock_guard<std::mutex> lock(fmt_.monlock_);
    exception_detected_ = true;
  }

  void FastMonitoringService::setExceptionDetected(unsigned int ls) {
    if (!ls)
      exception_detected_ = true;
    else
      exceptionInLS_.push_back(ls);
  }

  void FastMonitoringService::jobFailure() { macrostate_ = FastMonitoringThread::sError; }

  //new output module name is stream
  void FastMonitoringService::preModuleBeginJob(const edm::ModuleDescription& desc) {
    std::lock_guard<std::mutex> lock(fmt_.monlock_);
    //std::cout << " Pre module Begin Job module: " << desc.moduleName() << std::endl;

    //build a map of modules keyed by their module description address
    //here we need to treat output modules in a special way so they can be easily singled out
    if (desc.moduleName() == "Stream" || desc.moduleName() == "ShmStreamConsumer" ||
        desc.moduleName() == "EvFOutputModule" || desc.moduleName() == "EventStreamFileWriter" ||
        desc.moduleName() == "PoolOutputModule") {
      encModule_.updateReserved((void*)&desc);
      nOutputModules_++;
    } else
      encModule_.update((void*)&desc);
  }

  void FastMonitoringService::postBeginJob() {
    std::string&& moduleLegStrJson = makeModuleLegendaJson();
    FileIO::writeStringToFile(moduleLegendFileJson_, moduleLegStrJson);

    std::string inputLegendStrJson = makeInputLegendaJson();
    FileIO::writeStringToFile(inputLegendFileJson_, inputLegendStrJson);

    macrostate_ = FastMonitoringThread::sJobReady;

    //update number of entries in module histogram
    std::lock_guard<std::mutex> lock(fmt_.monlock_);
    fmt_.m_data.microstateBins_ = encModule_.vecsize();
  }

  void FastMonitoringService::postEndJob() {
    macrostate_ = FastMonitoringThread::sJobEnded;
    fmt_.stop();
  }

  void FastMonitoringService::postGlobalBeginRun(edm::GlobalContext const& gc) {
    macrostate_ = FastMonitoringThread::sRunning;
    isInitTransition_ = false;
  }

  void FastMonitoringService::preGlobalBeginLumi(edm::GlobalContext const& gc) {
    timeval lumiStartTime;
    gettimeofday(&lumiStartTime, nullptr);
    unsigned int newLumi = gc.luminosityBlockID().luminosityBlock();
    lastGlobalLumi_ = newLumi;

    std::lock_guard<std::mutex> lock(fmt_.monlock_);
    lumiStartTime_[newLumi] = lumiStartTime;
  }

  void FastMonitoringService::preGlobalEndLumi(edm::GlobalContext const& gc) {
    unsigned int lumi = gc.luminosityBlockID().luminosityBlock();
    LogDebug("FastMonitoringService") << "Lumi ended. Writing JSON information. LUMI -: " << lumi;
    timeval lumiStopTime;
    gettimeofday(&lumiStopTime, nullptr);

    std::lock_guard<std::mutex> lock(fmt_.monlock_);

    // Compute throughput
    timeval stt = lumiStartTime_[lumi];
    lumiStartTime_.erase(lumi);
    unsigned long usecondsForLumi = (lumiStopTime.tv_sec - stt.tv_sec) * 1000000 + (lumiStopTime.tv_usec - stt.tv_usec);
    unsigned long accuSize = accuSize_.find(lumi) == accuSize_.end() ? 0 : accuSize_[lumi];
    accuSize_.erase(lumi);
    double throughput = throughputFactor() * double(accuSize) / double(usecondsForLumi);
    //store to registered variable
    fmt_.m_data.fastThroughputJ_.value() = throughput;

    //update
    doSnapshot(lumi, true);

    //retrieve one result we need (todo: sanity check if it's found)
    IntJ* lumiProcessedJptr = dynamic_cast<IntJ*>(fmt_.jsonMonitor_->getMergedIntJForLumi("Processed", lumi));
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

    if (inputSource_) {
      auto sourceReport = inputSource_->getEventReport(lumi, true);
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
    if (filePerFwkStream_) {
      std::stringstream slowFileNameStem;
      slowFileNameStem << slowName_ << "_ls" << std::setfill('0') << std::setw(4) << lumi << "_pid" << std::setfill('0')
                       << std::setw(5) << getpid();
      boost::filesystem::path slow = workingDirectory_;
      slow /= slowFileNameStem.str();
      fmt_.jsonMonitor_->outputFullJSONs(slow.string(), ".jsn", lumi);
    } else {
      std::stringstream slowFileName;
      slowFileName << slowName_ << "_ls" << std::setfill('0') << std::setw(4) << lumi << "_pid" << std::setfill('0')
                   << std::setw(5) << getpid() << ".jsn";
      boost::filesystem::path slow = workingDirectory_;
      slow /= slowFileName.str();
      fmt_.jsonMonitor_->outputFullJSON(slow.string(),
                                        lumi);  //full global and stream merge and JSON write for this lumi
    }
    fmt_.jsonMonitor_->discardCollected(lumi);  //we don't do further updates for this lumi
  }

  void FastMonitoringService::postGlobalEndLumi(edm::GlobalContext const& gc) {
    std::lock_guard<std::mutex> lock(fmt_.monlock_);
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

    std::lock_guard<std::mutex> lock(fmt_.monlock_);
    fmt_.m_data.streamLumi_[sid] = sc.eventID().luminosityBlock();

    //reset collected values for this stream
    *(fmt_.m_data.processed_[sid]) = 0;

    ministate_[sid] = &nopath_;
    microstate_[sid] = &reservedMicroStateNames[mBoL];
  }

  void FastMonitoringService::postStreamBeginLumi(edm::StreamContext const& sc) {
    microstate_[sc.streamID().value()] = &reservedMicroStateNames[mIdle];
  }

  void FastMonitoringService::preStreamEndLumi(edm::StreamContext const& sc) {
    unsigned int sid = sc.streamID().value();
    std::lock_guard<std::mutex> lock(fmt_.monlock_);

    //update processed count to be complete at this time
    doStreamEOLSnapshot(sc.eventID().luminosityBlock(), sid);
    //reset this in case stream does not get notified of next lumi (we keep processed events only)
    ministate_[sid] = &nopath_;
    microstate_[sid] = &reservedMicroStateNames[mEoL];
  }
  void FastMonitoringService::postStreamEndLumi(edm::StreamContext const& sc) {
    microstate_[sc.streamID().value()] = &reservedMicroStateNames[mFwkEoL];
  }

  void FastMonitoringService::prePathEvent(edm::StreamContext const& sc, edm::PathContext const& pc) {
    //make sure that all path names are retrieved before allowing ministate to change
    //hack: assume memory is synchronized after ~50 events seen by each stream
    if (UNLIKELY(eventCountForPathInit_[sc.streamID()] < 50) &&
        false == collectedPathList_[sc.streamID()]->load(std::memory_order_acquire)) {
      //protection between stream threads, as well as the service monitoring thread
      std::lock_guard<std::mutex> lock(fmt_.monlock_);

      if (firstEventId_[sc.streamID()] == 0)
        firstEventId_[sc.streamID()] = sc.eventID().event();
      if (sc.eventID().event() == firstEventId_[sc.streamID()]) {
        encPath_[sc.streamID()].update((void*)&pc.pathName());
        return;
      } else {
        //finished collecting path names
        collectedPathList_[sc.streamID()]->store(true, std::memory_order_seq_cst);
        fmt_.m_data.ministateBins_ = encPath_[sc.streamID()].vecsize();
        if (!pathLegendWritten_) {
          std::string pathLegendStrJson = makePathLegendaJson();
          FileIO::writeStringToFile(pathLegendFileJson_, pathLegendStrJson);
          pathLegendWritten_ = true;
        }
      }
    } else {
      ministate_[sc.streamID()] = &(pc.pathName());
    }
  }

  void FastMonitoringService::preEvent(edm::StreamContext const& sc) {}

  void FastMonitoringService::postEvent(edm::StreamContext const& sc) {
    microstate_[sc.streamID()] = &reservedMicroStateNames[mIdle];

    ministate_[sc.streamID()] = &nopath_;

    (*(fmt_.m_data.processed_[sc.streamID()]))++;
    eventCountForPathInit_[sc.streamID()].m_value++;

    //fast path counter (events accumulated in a run)
    unsigned long res = totalEventsProcessed_.fetch_add(1, std::memory_order_relaxed);
    fmt_.m_data.fastPathProcessedJ_ = res + 1;
    //fmt_.m_data.fastPathProcessedJ_ = totalEventsProcessed_.load(std::memory_order_relaxed);
  }

  void FastMonitoringService::preSourceEvent(edm::StreamID sid) {
    microstate_[sid.value()] = &reservedMicroStateNames[mInput];
  }

  void FastMonitoringService::postSourceEvent(edm::StreamID sid) {
    microstate_[sid.value()] = &reservedMicroStateNames[mFwkOvhSrc];
  }

  void FastMonitoringService::preModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
    microstate_[sc.streamID().value()] = (void*)(mcc.moduleDescription());
  }

  void FastMonitoringService::postModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
    //microstate_[sc.streamID().value()] = (void*)(mcc.moduleDescription());
    microstate_[sc.streamID().value()] = &reservedMicroStateNames[mFwkOvhMod];
  }

  //FUNCTIONS CALLED FROM OUTSIDE

  //this is for old-fashioned service that is not thread safe and can block other streams
  //(we assume the worst case - everything is blocked)
  void FastMonitoringService::setMicroState(MicroStateService::Microstate m) {
    for (unsigned int i = 0; i < nStreams_; i++)
      microstate_[i] = &reservedMicroStateNames[m];
  }

  //this is for services that are multithreading-enabled or rarely blocks other streams
  void FastMonitoringService::setMicroState(edm::StreamID sid, MicroStateService::Microstate m) {
    microstate_[sid] = &reservedMicroStateNames[m];
  }

  //from source
  void FastMonitoringService::accumulateFileSize(unsigned int lumi, unsigned long fileSize) {
    std::lock_guard<std::mutex> lock(fmt_.monlock_);

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
    std::lock_guard<std::mutex> lock(fmt_.monlock_);

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
    std::lock_guard<std::mutex> lock(fmt_.monlock_);
    lockStatsDuringLumi_[ls] = std::pair<double, unsigned int>(waitTime, lockCount);
  }

  //for the output module
  unsigned int FastMonitoringService::getEventsProcessedForLumi(unsigned int lumi, bool* abortFlag) {
    std::lock_guard<std::mutex> lock(fmt_.monlock_);

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
    std::lock_guard<std::mutex> lock(fmt_.monlock_);

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

  void FastMonitoringService::doSnapshot(const unsigned int ls, const bool isGlobalEOL) {
    // update macrostate
    fmt_.m_data.fastMacrostateJ_ = macrostate_;

    std::vector<const void*> microstateCopy(microstate_.begin(), microstate_.end());

    if (!isInitTransition_) {
      auto itd = avgLeadTime_.find(ls);
      if (itd != avgLeadTime_.end())
        fmt_.m_data.fastAvgLeadTimeJ_ = itd->second;
      else
        fmt_.m_data.fastAvgLeadTimeJ_ = 0.;

      auto iti = filesProcessedDuringLumi_.find(ls);
      if (iti != filesProcessedDuringLumi_.end())
        fmt_.m_data.fastFilesProcessedJ_ = iti->second;
      else
        fmt_.m_data.fastFilesProcessedJ_ = 0;

      auto itrd = lockStatsDuringLumi_.find(ls);
      if (itrd != lockStatsDuringLumi_.end()) {
        fmt_.m_data.fastLockWaitJ_ = itrd->second.first;
        fmt_.m_data.fastLockCountJ_ = itrd->second.second;
      } else {
        fmt_.m_data.fastLockWaitJ_ = 0.;
        fmt_.m_data.fastLockCountJ_ = 0.;
      }
    }

    for (unsigned int i = 0; i < nStreams_; i++) {
      fmt_.m_data.ministateEncoded_[i] = encPath_[i].encode(ministate_[i]);
      fmt_.m_data.microstateEncoded_[i] = encModule_.encode(microstateCopy[i]);
    }

    bool inputStatePerThread = false;

    if (inputState_ == FastMonitoringThread::inWaitInput) {
      switch (inputSupervisorState_) {
        case FastMonitoringThread::inSupFileLimit:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_fileLimit;
          break;
        case FastMonitoringThread::inSupWaitFreeChunk:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_waitFreeChunk;
          break;
        case FastMonitoringThread::inSupWaitFreeChunkCopying:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_waitFreeChunkCopying;
          break;
        case FastMonitoringThread::inSupWaitFreeThread:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_waitFreeThread;
          break;
        case FastMonitoringThread::inSupWaitFreeThreadCopying:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_waitFreeThreadCopying;
          break;
        case FastMonitoringThread::inSupBusy:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_busy;
          break;
        case FastMonitoringThread::inSupLockPolling:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_lockPolling;
          break;
        case FastMonitoringThread::inSupLockPollingCopying:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_lockPollingCopying;
          break;
        case FastMonitoringThread::inRunEnd:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_runEnd;
          break;
        case FastMonitoringThread::inSupNoFile:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_noFile;
          break;
        case FastMonitoringThread::inSupNewFile:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_newFile;
          break;
        case FastMonitoringThread::inSupNewFileWaitThreadCopying:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_newFileWaitThreadCopying;
          break;
        case FastMonitoringThread::inSupNewFileWaitThread:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_newFileWaitThread;
          break;
        case FastMonitoringThread::inSupNewFileWaitChunkCopying:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_newFileWaitChunkCopying;
          break;
        case FastMonitoringThread::inSupNewFileWaitChunk:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput_newFileWaitChunk;
          break;
        default:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitInput;
      }
    } else if (inputState_ == FastMonitoringThread::inWaitChunk) {
      switch (inputSupervisorState_) {
        case FastMonitoringThread::inSupFileLimit:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_fileLimit;
          break;
        case FastMonitoringThread::inSupWaitFreeChunk:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_waitFreeChunk;
          break;
        case FastMonitoringThread::inSupWaitFreeChunkCopying:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_waitFreeChunkCopying;
          break;
        case FastMonitoringThread::inSupWaitFreeThread:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_waitFreeThread;
          break;
        case FastMonitoringThread::inSupWaitFreeThreadCopying:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_waitFreeThreadCopying;
          break;
        case FastMonitoringThread::inSupBusy:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_busy;
          break;
        case FastMonitoringThread::inSupLockPolling:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_lockPolling;
          break;
        case FastMonitoringThread::inSupLockPollingCopying:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_lockPollingCopying;
          break;
        case FastMonitoringThread::inRunEnd:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_runEnd;
          break;
        case FastMonitoringThread::inSupNoFile:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_noFile;
          break;
        case FastMonitoringThread::inSupNewFile:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_newFile;
          break;
        case FastMonitoringThread::inSupNewFileWaitThreadCopying:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_newFileWaitThreadCopying;
          break;
        case FastMonitoringThread::inSupNewFileWaitThread:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_newFileWaitThread;
          break;
        case FastMonitoringThread::inSupNewFileWaitChunkCopying:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_newFileWaitChunkCopying;
          break;
        case FastMonitoringThread::inSupNewFileWaitChunk:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk_newFileWaitChunk;
          break;
        default:
          fmt_.m_data.inputState_[0] = FastMonitoringThread::inWaitChunk;
      }
    } else if (inputState_ == FastMonitoringThread::inNoRequest) {
      inputStatePerThread = true;
      for (unsigned int i = 0; i < nStreams_; i++) {
        if (microstateCopy[i] == &reservedMicroStateNames[mIdle])
          fmt_.m_data.inputState_[i] = FastMonitoringThread::inNoRequestWithIdleThreads;
        else if (microstateCopy[i] == &reservedMicroStateNames[mEoL] ||
                 microstateCopy[i] == &reservedMicroStateNames[mFwkEoL])
          fmt_.m_data.inputState_[i] = FastMonitoringThread::inNoRequestWithEoLThreads;
        else
          fmt_.m_data.inputState_[i] = FastMonitoringThread::inNoRequest;
      }
    } else if (inputState_ == FastMonitoringThread::inNewLumi) {
      inputStatePerThread = true;
      for (unsigned int i = 0; i < nStreams_; i++) {
        if (microstateCopy[i] == &reservedMicroStateNames[mEoL] ||
            microstateCopy[i] == &reservedMicroStateNames[mFwkEoL])
          fmt_.m_data.inputState_[i] = FastMonitoringThread::inNewLumi;
      }
    } else
      fmt_.m_data.inputState_[0] = inputState_;

    //this is same for all streams
    if (!inputStatePerThread)
      for (unsigned int i = 1; i < nStreams_; i++)
        fmt_.m_data.inputState_[i] = fmt_.m_data.inputState_[0];

    if (isGlobalEOL) {  //only update global variables
      fmt_.jsonMonitor_->snapGlobal(ls);
    } else
      fmt_.jsonMonitor_->snap(ls);
  }

}  //end namespace evf
