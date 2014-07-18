#include "EventFilter/Utilities/plugins/FastMonitoringService.h"
#include <iostream>

#include "FWCore/Framework/interface/Event.h"
#include <iomanip>
#include <sys/time.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "EventFilter/Utilities/plugins/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

constexpr double throughputFactor() {return (1000000)/double(1024*1024);}

namespace evf{

  const std::string FastMonitoringService::macroStateNames[FastMonitoringThread::MCOUNT] = 
    {"Init","JobReady","RunGiven","Running",
     "Stopping","Done","JobEnded","Error","ErrorEnded","End",
     "Invalid"};

  const std::string FastMonitoringService::nopath_ = "NoPath";

  FastMonitoringService::FastMonitoringService(const edm::ParameterSet& iPS, 
				       edm::ActivityRegistry& reg) : 
    MicroStateService(iPS,reg)
    ,encModule_(33)
    ,nStreams_(0)//until initialized
    ,sleepTime_(iPS.getUntrackedParameter<int>("sleepTime", 1))
    ,fastMonIntervals_(iPS.getUntrackedParameter<unsigned int>("fastMonIntervals", 1))
    ,microstateDefPath_(iPS.getUntrackedParameter<std::string> ("microstateDefPath", std::string(getenv("CMSSW_BASE"))+"/src/EventFilter/Utilities/plugins/microstatedef.jsd"))
    ,fastMicrostateDefPath_(iPS.getUntrackedParameter<std::string>("fastMicrostateDefPath", microstateDefPath_))
    ,fastName_(iPS.getUntrackedParameter<std::string>("fastName", "fastmoni"))
    ,slowName_(iPS.getUntrackedParameter<std::string>("slowName", "slowmoni"))
    ,totalEventsProcessed_(0)
  {
    reg.watchPreallocate(this, &FastMonitoringService::preallocate);//receiving information on number of threads
    reg.watchJobFailure(this,&FastMonitoringService::jobFailure);//global

    reg.watchPreModuleBeginJob(this,&FastMonitoringService::preModuleBeginJob);//global
    reg.watchPostBeginJob(this,&FastMonitoringService::postBeginJob);
    reg.watchPostEndJob(this,&FastMonitoringService::postEndJob);

    reg.watchPreGlobalBeginLumi(this,&FastMonitoringService::preGlobalBeginLumi);//global lumi
    reg.watchPreGlobalEndLumi(this,&FastMonitoringService::preGlobalEndLumi);
    reg.watchPostGlobalEndLumi(this,&FastMonitoringService::postGlobalEndLumi);

    reg.watchPreStreamBeginLumi(this,&FastMonitoringService::preStreamBeginLumi);//stream lumi
    reg.watchPostStreamBeginLumi(this,&FastMonitoringService::postStreamBeginLumi);
    reg.watchPreStreamEndLumi(this,&FastMonitoringService::preStreamEndLumi);
    reg.watchPostStreamEndLumi(this,&FastMonitoringService::postStreamEndLumi);

    reg.watchPrePathEvent(this,&FastMonitoringService::prePathEvent);

    reg.watchPreEvent(this,&FastMonitoringService::preEvent);//stream
    reg.watchPostEvent(this,&FastMonitoringService::postEvent);

    reg.watchPreSourceEvent(this,&FastMonitoringService::preSourceEvent);//source (with streamID of requestor)
    reg.watchPostSourceEvent(this,&FastMonitoringService::postSourceEvent);
  
    reg.watchPreModuleEvent(this,&FastMonitoringService::preModuleEvent);//should be stream
    reg.watchPostModuleEvent(this,&FastMonitoringService::postModuleEvent);//

    edm::LogInfo("FastMonitoringService") << "Constructed";
  }


  FastMonitoringService::~FastMonitoringService()
  {
  }


  std::string FastMonitoringService::makePathLegenda(){
    //taken from first stream
    std::ostringstream ost;
    for(int i = 0;
	i < encPath_[0].current_;
	i++)
      ost<<i<<"="<<*((std::string *)(encPath_[0].decode(i)))<<" ";
    return ost.str();
  }


  std::string FastMonitoringService::makeModuleLegenda()
  {  
    std::ostringstream ost;
    for(int i = 0; i < encModule_.current_; i++)
      ost<<i<<"="<<((const edm::ModuleDescription *)(encModule_.decode(i)))->moduleLabel()<<" ";
    return ost.str();
  }


  void FastMonitoringService::preallocate(edm::service::SystemBounds const & bounds)
  {

    // FIND RUN DIRECTORY
    // The run dir should be set via the configuration of EvFDaqDirector
    
    if (edm::Service<evf::EvFDaqDirector>().operator->()==nullptr)
    {
      throw cms::Exception("FastMonitoringService") << "ERROR: EvFDaqDirector is not present";
    
    }
    boost::filesystem::path runDirectory(edm::Service<evf::EvFDaqDirector>()->findCurrentRunDir());
    workingDirectory_ = runDirectory_ = runDirectory;
    workingDirectory_ /= "mon";

    if ( !boost::filesystem::is_directory(workingDirectory_)) {
        edm::LogInfo("FastMonitoringService") << "<MON> DIR NOT FOUND! Trying to create " << workingDirectory_.string() ;
        boost::filesystem::create_directories(workingDirectory_);
        if ( !boost::filesystem::is_directory(workingDirectory_))
          edm::LogWarning("FastMonitoringService") << "Unable to create <MON> DIR " << workingDirectory_.string()
                                                   << ". No monitoring data will be written.";
    }

    std::ostringstream fastFileName;

    fastFileName << fastName_ << "_pid" << std::setfill('0') << std::setw(5) << getpid() << ".fast";
    boost::filesystem::path fast = workingDirectory_;
    fast /= fastFileName.str();
    fastPath_ = fast.string();

    std::ostringstream moduleLegFile;
    moduleLegFile << "microstatelegend_pid" << std::setfill('0') << std::setw(5) << getpid() << ".leg";
    moduleLegendFile_  = (workingDirectory_/moduleLegFile.str()).string();

    std::ostringstream pathLegFile;
    pathLegFile << "pathlegend_pid" << std::setfill('0') << std::setw(5) << getpid() << ".leg";
    pathLegendFile_  = (workingDirectory_/pathLegFile.str()).string();

    edm::LogInfo("FastMonitoringService") << "preallocate: initializing FastMonitor with microstate def path: "
			                  << microstateDefPath_ << " " << FastMonitoringThread::MCOUNT << " ";
			                  //<< encPath_.current_ + 1 << " " << encModule_.current_ + 1

    nStreams_=bounds.maxNumberOfStreams();
    nThreads_=bounds.maxNumberOfThreads();

    //this should already be >=1
    if (nStreams_==0) nStreams_=1;
    if (nThreads_==0) nThreads_=1;

    /*
     * initialize the fast monitor with:
     * vector of pointers to monitorable parameters
     * path to definition
     *
     */

    macrostate_=FastMonitoringThread::sInit;

    for(unsigned int i = 0; i < (mCOUNT); i++)
      encModule_.updateReserved((void*)(reservedMicroStateNames+i));
    encModule_.completeReservedWithDummies();

    for (unsigned int i=0;i<nStreams_;i++) {
       ministate_.push_back(&nopath_);
       microstate_.push_back(&reservedMicroStateNames[mInvalid]);

       //for synchronization
       streamCounterUpdating_.push_back(new std::atomic<bool>(0));

       //path (mini) state
       encPath_.emplace_back(0);
       encPath_[i].update((void*)&nopath_);
       eventCountForPathInit_.push_back(0);
       firstEventId_.push_back(0);
       collectedPathList_.push_back(new std::atomic<bool>(0));

    }
    //for (unsigned int i=0;i<nThreads_;i++)
    //  threadMicrostate_.push_back(&reservedMicroStateNames[mInvalid]);

    //initial size until we detect number of bins
    fmt_.m_data.macrostateBins_=FastMonitoringThread::MCOUNT;
    fmt_.m_data.ministateBins_=0;
    fmt_.m_data.microstateBins_ = 0; 
 
    lastGlobalLumi_=0; 
    isGlobalLumiTransition_=true;
    lumiFromSource_=0;

    //startup monitoring
    fmt_.resetFastMonitor(microstateDefPath_,fastMicrostateDefPath_);
    fmt_.jsonMonitor_->setNStreams(nStreams_);
    fmt_.m_data.registerVariables(fmt_.jsonMonitor_.get(), nStreams_, threadIDAvailable_ ? nThreads_:0);
    monInit_.store(false,std::memory_order_release);
    fmt_.start(&FastMonitoringService::dowork,this);

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

  void FastMonitoringService::jobFailure()
  {
    macrostate_ = FastMonitoringThread::sError;
  }

  //new output module name is stream
  void FastMonitoringService::preModuleBeginJob(const edm::ModuleDescription& desc)
  {
    std::lock_guard<std::mutex> lock(fmt_.monlock_);
    //std::cout << " Pre module Begin Job module: " << desc.moduleName() << std::endl;

    //build a map of modules keyed by their module description address
    //here we need to treat output modules in a special way so they can be easily singled out
    if(desc.moduleName() == "Stream" || desc.moduleName() == "ShmStreamConsumer" ||
       desc.moduleName() == "EventStreamFileWriter" || desc.moduleName() == "PoolOutputModule")
      encModule_.updateReserved((void*)&desc);
    else
      encModule_.update((void*)&desc);
  }

  void FastMonitoringService::postBeginJob()
  {
    std::string && moduleLegStr = makeModuleLegenda();
    FileIO::writeStringToFile(moduleLegendFile_, moduleLegStr);

    macrostate_ = FastMonitoringThread::sJobReady;

    //update number of entries in module histogram
    std::lock_guard<std::mutex> lock(fmt_.monlock_);
    fmt_.m_data.microstateBins_ = encModule_.vecsize();
  }

  void FastMonitoringService::postEndJob()
  {
    macrostate_ = FastMonitoringThread::sJobEnded;
    fmt_.stop();
  }

  void FastMonitoringService::postGlobalBeginRun(edm::GlobalContext const& gc)
  {
    macrostate_ = FastMonitoringThread::sRunning;
  }

  void FastMonitoringService::preGlobalBeginLumi(edm::GlobalContext const& gc)
  {
          edm::LogInfo("FastMonitoringService") << gc.luminosityBlockID().luminosityBlock();

	  timeval lumiStartTime;
	  gettimeofday(&lumiStartTime, 0);
	  unsigned int newLumi = gc.luminosityBlockID().luminosityBlock();

          std::lock_guard<std::mutex> lock(fmt_.monlock_);

	  lumiStartTime_[newLumi]=lumiStartTime;
	  while (!lastGlobalLumisClosed_.empty()) {
		  //wipe out old map entries as they aren't needed and slow down access
		  unsigned int oldLumi = lastGlobalLumisClosed_.back();
		  lastGlobalLumisClosed_.pop();
		  lumiStartTime_.erase(oldLumi);
		  avgLeadTime_.erase(oldLumi);
		  filesProcessedDuringLumi_.erase(oldLumi);
		  accuSize_.erase(oldLumi);
		  processedEventsPerLumi_.erase(oldLumi);
	  }
	  lastGlobalLumi_= newLumi;
	  isGlobalLumiTransition_=false;
  }

  void FastMonitoringService::preGlobalEndLumi(edm::GlobalContext const& gc)
  {
	  unsigned int lumi = gc.luminosityBlockID().luminosityBlock();
          edm::LogInfo("FastMonitoringService") << "FastMonitoringService: LUMI: "
                                                << lumi << " ended! Writing JSON information...";
	  timeval lumiStopTime;
	  gettimeofday(&lumiStopTime, 0);

          std::lock_guard<std::mutex> lock(fmt_.monlock_);

	  // Compute throughput
	  timeval stt = lumiStartTime_[lumi];
	  unsigned long usecondsForLumi = (lumiStopTime.tv_sec - stt.tv_sec)*1000000
		                  + (lumiStopTime.tv_usec - stt.tv_usec);
	  unsigned long accuSize = accuSize_.find(lumi)==accuSize_.end() ? 0 : accuSize_[lumi];
	  double throughput = throughputFactor()* double(accuSize) / double(usecondsForLumi);
	  //store to registered variable
	  fmt_.m_data.fastThroughputJ_.value() = throughput;

	  //update
	  doSnapshot(lumi,true);

	  // create file name for slow monitoring file
	  std::stringstream slowFileName;
	  slowFileName << slowName_ << "_ls" << std::setfill('0') << std::setw(4)
			<< lumi << "_pid" << std::setfill('0')
			<< std::setw(5) << getpid() << ".jsn";
	  boost::filesystem::path slow = workingDirectory_;
	  slow /= slowFileName.str();

	  //retrieve one result we need (todo: sanity check if it's found)
	  IntJ *lumiProcessedJptr = dynamic_cast<IntJ*>(fmt_.jsonMonitor_->getMergedIntJForLumi("Processed",lumi));
          if (!lumiProcessedJptr)
              throw cms::Exception("FastMonitoringService") << "Internal error: got null pointer from FastMonitor";
	  processedEventsPerLumi_[lumi] = lumiProcessedJptr->value();

	  {
	    auto itr = sourceEventsReport_.find(lumi);
	    if (itr==sourceEventsReport_.end()) {
              throw cms::Exception("FastMonitoringService") << "ERROR: SOURCE did not send update for lumi block " << lumi;
	    }
	    else {
	      if (itr->second!=processedEventsPerLumi_[lumi]) {
		throw cms::Exception("FastMonitoringService") << " ERROR: MISMATCH with SOURCE update for lumi block" << lumi
                                                              << ", events(processed):" << processedEventsPerLumi_[lumi]
                                                              << " events(source):" << itr->second;
	      }
	      sourceEventsReport_.erase(itr);
	    }
	  }
	  edm::LogInfo("FastMonitoringService")	<< ">>> >>> FastMonitoringService: processed event count for this lumi = "
			                        << lumiProcessedJptr->value() << " time = " << usecondsForLumi/1000000
			                        << " size = " << accuSize << " thr = " << throughput;
	  delete lumiProcessedJptr;

	  //full global and stream merge&output for this lumi
	  fmt_.jsonMonitor_->outputFullJSON(slow.string(),lumi);//full global and stream merge and JSON write for this lumi
	  fmt_.jsonMonitor_->discardCollected(lumi);//we don't do further updates for this lumi

	  isGlobalLumiTransition_=true;
  }

  void FastMonitoringService::postGlobalEndLumi(edm::GlobalContext const& gc)
  {
    edm::LogInfo("FastMonitoringService") << "FastMonitoringService: Post-global-end LUMI: "
                                          << gc.luminosityBlockID().luminosityBlock();
    //mark closed lumis (still keep map entries until next one)
    lastGlobalLumisClosed_.push(gc.luminosityBlockID().luminosityBlock());
  }

  void FastMonitoringService::preStreamBeginLumi(edm::StreamContext const& sc)
  {
    unsigned int sid = sc.streamID().value();
    std::lock_guard<std::mutex> lock(fmt_.monlock_);
    fmt_.m_data.streamLumi_[sid] = sc.eventID().luminosityBlock();

    //reset collected values for this stream
    *(fmt_.m_data.processed_[sid])=0;

    ministate_[sid]=&nopath_;
    microstate_[sid]=&reservedMicroStateNames[mFwkOvh];
  }

  void FastMonitoringService::postStreamBeginLumi(edm::StreamContext const& sc)
  {
    microstate_[sc.streamID().value()]=&reservedMicroStateNames[mIdle];
  }

  void FastMonitoringService::preStreamEndLumi(edm::StreamContext const& sc)
  {
    unsigned int sid = sc.streamID().value();
    std::lock_guard<std::mutex> lock(fmt_.monlock_);

    #if ATOMIC_LEVEL>=2
    //spinlock to make sure we are not still updating event counter somewhere
    while (streamCounterUpdating_[sid]->load(std::memory_order_acquire)) {}
    #endif

    //update processed count to be complete at this time
    doStreamEOLSnapshot(sc.eventID().luminosityBlock(),sid);
    //reset this in case stream does not get notified of next lumi (we keep processed events only)
    ministate_[sid]=&nopath_;
    microstate_[sid]=&reservedMicroStateNames[mEoL];
  }
  void FastMonitoringService::postStreamEndLumi(edm::StreamContext const& sc)
  {
    microstate_[sc.streamID().value()]=&reservedMicroStateNames[mFwkOvh];
  }


  void FastMonitoringService::prePathEvent(edm::StreamContext const& sc, edm::PathContext const& pc)
  {
    //make sure that all path names are retrieved before allowing ministate to change
    //hack: assume memory is synchronized after ~50 events seen by each stream
    if (unlikely(eventCountForPathInit_[sc.streamID()]<50) && false==collectedPathList_[sc.streamID()]->load(std::memory_order_acquire))
    {
      //protection between stream threads, as well as the service monitoring thread
      std::lock_guard<std::mutex> lock(fmt_.monlock_);

      if (firstEventId_[sc.streamID()]==0) 
	firstEventId_[sc.streamID()]=sc.eventID().event();
      if (sc.eventID().event()==firstEventId_[sc.streamID()])
      {
	encPath_[sc.streamID()].update((void*)&pc.pathName());
	return;
      }
      else {
	//finished collecting path names
	collectedPathList_[sc.streamID()]->store(true,std::memory_order_seq_cst);
	fmt_.m_data.ministateBins_=encPath_[sc.streamID()].vecsize();
	if (!pathLegendWritten_) {
	  std::string pathLegendStr =  makePathLegenda();
	  FileIO::writeStringToFile(pathLegendFile_, pathLegendStr);
	  pathLegendWritten_=true;
	}
      }
    }
    else {
      ministate_[sc.streamID()] = &(pc.pathName());
    }
  }


  void FastMonitoringService::preEvent(edm::StreamContext const& sc)
  {
  }

  void FastMonitoringService::postEvent(edm::StreamContext const& sc)
  {
    microstate_[sc.streamID()] = &reservedMicroStateNames[mIdle];

    ministate_[sc.streamID()] = &nopath_;

    #if ATOMIC_LEVEL>=2
    //use atomic flag to make sure end of lumi sees this
    streamCounterUpdating_[sc.streamID()]->store(true,std::memory_order_release);
    fmt_.m_data.processed_[sc.streamID()]->fetch_add(1,std::memory_order_release);
    streamCounterUpdating_[sc.streamID()]->store(false,std::memory_order_release);

    #elif ATOMIC_LEVEL==1
    //writes are atomic, we assume writes propagate to memory before stream EOL snap
    fmt_.m_data.processed_[sc.streamID()]->fetch_add(1,std::memory_order_relaxed);

    #elif ATOMIC_LEVEL==0 //default
    (*(fmt_.m_data.processed_[sc.streamID()]))++;
    #endif
    eventCountForPathInit_[sc.streamID()]++;

    //fast path counter (events accumulated in a run)
    unsigned long res = totalEventsProcessed_.fetch_add(1,std::memory_order_relaxed);
    fmt_.m_data.fastPathProcessedJ_ = res+1; 
    //fmt_.m_data.fastPathProcessedJ_ = totalEventsProcessed_.load(std::memory_order_relaxed);
  }

  void FastMonitoringService::preSourceEvent(edm::StreamID sid)
  {
    microstate_[sid.value()] = &reservedMicroStateNames[mInput];
  }

  void FastMonitoringService::postSourceEvent(edm::StreamID sid)
  {
    microstate_[sid.value()] = &reservedMicroStateNames[mFwkOvh];
  }

  void FastMonitoringService::preModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc)
  {
    microstate_[sc.streamID().value()] = (void*)(mcc.moduleDescription());
  }

  void FastMonitoringService::postModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc)
  {
    //microstate_[sc.streamID().value()] = (void*)(mcc.moduleDescription());
    microstate_[sc.streamID().value()] = &reservedMicroStateNames[mFwkOvh];
  }

  //FUNCTIONS CALLED FROM OUTSIDE

  //this is for old-fashioned service that is not thread safe and can block other streams
  //(we assume the worst case - everything is blocked)
  void FastMonitoringService::setMicroState(MicroStateService::Microstate m)
  {
    for (unsigned int i=0;i<nStreams_;i++)
      microstate_[i] = &reservedMicroStateNames[m];
  }

  //this is for services that are multithreading-enabled or rarely blocks other streams
  void FastMonitoringService::setMicroState(edm::StreamID sid, MicroStateService::Microstate m)
  {
    microstate_[sid] = &reservedMicroStateNames[m];
  }

  //from source
  void FastMonitoringService::accumulateFileSize(unsigned int lumi, unsigned long fileSize) {
        std::lock_guard<std::mutex> lock(fmt_.monlock_);

	if (accuSize_.find(lumi)==accuSize_.end()) accuSize_[lumi] = fileSize;
	else accuSize_[lumi] += fileSize;

	if (filesProcessedDuringLumi_.find(lumi)==filesProcessedDuringLumi_.end())
	  filesProcessedDuringLumi_[lumi] = 1;
	else
	  filesProcessedDuringLumi_[lumi]++;
  }

  void FastMonitoringService::startedLookingForFile() {
	  gettimeofday(&fileLookStart_, 0);
	  /*
	 std::cout << "Started looking for .raw file at: s=" << fileLookStart_.tv_sec << ": ms = "
	 << fileLookStart_.tv_usec / 1000.0 << std::endl;
	 */
  }

  void FastMonitoringService::stoppedLookingForFile(unsigned int lumi) {
	  gettimeofday(&fileLookStop_, 0);
	  /*
	 std::cout << "Stopped looking for .raw file at: s=" << fileLookStop_.tv_sec << ": ms = "
	 << fileLookStop_.tv_usec / 1000.0 << std::endl;
	 */
          std::lock_guard<std::mutex> lock(fmt_.monlock_);

	  if (lumi>lumiFromSource_) {
		  lumiFromSource_=lumi;
		  leadTimes_.clear();
	  }
	  unsigned long elapsedTime = (fileLookStop_.tv_sec - fileLookStart_.tv_sec) * 1000000 // sec to us
	                              + (fileLookStop_.tv_usec - fileLookStart_.tv_usec); // us
	  // add this to lead times for this lumi
	  leadTimes_.push_back((double)elapsedTime);

	  // recompute average lead time for this lumi
	  if (leadTimes_.size() == 1) avgLeadTime_[lumi] = leadTimes_[0];
	  else {
		  double totTime = 0;
		  for (unsigned int i = 0; i < leadTimes_.size(); i++) totTime += leadTimes_[i];
		  avgLeadTime_[lumi] = 0.001*(totTime / leadTimes_.size());
	  }
  }

  //for the output module
  unsigned int FastMonitoringService::getEventsProcessedForLumi(unsigned int lumi) {
    std::lock_guard<std::mutex> lock(fmt_.monlock_);

    auto it = processedEventsPerLumi_.find(lumi);
    if (it!=processedEventsPerLumi_.end()) {
      unsigned int proc = it->second;
      return proc;
    }
    else {
      throw cms::Exception("FastMonitoringService") << "ERROR: output module wants deleted info ";
      return 0;
    }
  }


  void FastMonitoringService::doSnapshot(const unsigned int ls, const bool isGlobalEOL) {
    // update macrostate
    fmt_.m_data.fastMacrostateJ_ = macrostate_;

    //update these unless in the midst of a global transition
    if (!isGlobalLumiTransition_) {

      auto itd = avgLeadTime_.find(ls);
      if (itd != avgLeadTime_.end()) 
	fmt_.m_data.fastAvgLeadTimeJ_ = itd->second;
      else fmt_.m_data.fastAvgLeadTimeJ_=0.;

      auto iti = filesProcessedDuringLumi_.find(ls);
      if (iti != filesProcessedDuringLumi_.end())
	fmt_.m_data.fastFilesProcessedJ_ = iti->second;
      else fmt_.m_data.fastFilesProcessedJ_=0;
    }
    else return;

    //capture latest mini/microstate of streams
    for (unsigned int i=0;i<nStreams_;i++) {
      fmt_.m_data.ministateEncoded_[i] = encPath_[i].encode(ministate_[i]);
      fmt_.m_data.microstateEncoded_[i] = encModule_.encode(microstate_[i]);
    }
    //for (unsigned int i=0;i<nThreads_;i++)
    //  fmt_.m_data.threadMicrostateEncoded_[i] = encModule_.encode(threadMicrostate_[i]);
    
    if (isGlobalEOL)
    {//only update global variables
      fmt_.jsonMonitor_->snapGlobal(ls);
    }
    else
      fmt_.jsonMonitor_->snap(ls);
  }

  void FastMonitoringService::reportEventsThisLumiInSource(unsigned int lumi,unsigned int events)
  {

    std::lock_guard<std::mutex> lock(fmt_.monlock_);
    auto itr = sourceEventsReport_.find(lumi);
    if (itr!=sourceEventsReport_.end())
      itr->second+=events;
    else 
      sourceEventsReport_[lumi]=events;

  }
} //end namespace evf

