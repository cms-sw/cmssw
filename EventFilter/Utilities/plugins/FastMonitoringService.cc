#include "FastMonitoringService.h"
#include <iostream>

#include "FWCore/Framework/interface/Event.h"
#include <iomanip>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "EvFDaqDirector.h"

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
    ,encPath_(0)
    ,nStreams_(0)//until initialized
    ,sleepTime_(iPS.getUntrackedParameter<int>("sleepTime", 1))
    //,rootDirectory_(iPS.getUntrackedParameter<std::string>("rootDirectory", "/data"))
    ,microstateDefPath_(iPS.getUntrackedParameter<std::string>("microstateDefPath", "/tmp/def.jsd"))
    ,outputDefPath_(iPS.getUntrackedParameter<std::string>("outputDefPath", "/tmp/def.jsd"))
    ,fastName_(iPS.getUntrackedParameter<std::string>("fastName", "states"))
    ,slowName_(iPS.getUntrackedParameter<std::string>("slowName", "lumi"))
  {
    registry.watchPreallocate(this, &FastMonitoringService::preallocate);//receiving information on number of threads
    reg.watchJobFailure(this,&FastMonitoringService::jobFailure);//global

    reg.watchPreModuleBeginJob(this,&FastMonitoringService::preModuleBeginJob);//global
    reg.watchPostBeginJob(this,&FastMonitoringService::postBeginJob);
    reg.watchPostEndJob(this,&FastMonitoringService::postEndJob);

    //TODO:
    //reg.watchPrePathBeginRun(this,&FastMonitoringService::prePathBeginRun);//there is no equivalent of this (will need to get it at first event..)
    //reg.watchPostGlobalBeginRun(this,&FastMonitoringService::postBeginRun);

    reg.watchPreGlobalBeginLumi(this,&FastMonitoringService::preGlobalBeginLumi);//global lumi
    reg.watchPreGlobalEndLumi(this,&FastMonitoringService::preGlobalEndLumi);
    reg.watchPostGlobalEndLumi(this,&FastMonitoringService::postGlobalEndLumi);

    reg.watchPreStreamBeginLumi(this,&FastMonitoringService::preStreamBeginLumi);//stream lumi
    reg.watchPreStreamEndLumi(this,&FastMonitoringService::preStreamEndLumi);

    reg.watchPreProcessPath(this,&FastMonitoringService::preProcessPath);//?

    reg.watchPreEvent(this,&FastMonitoringService::preEvent);//stream
    reg.watchPostEvent(this,&FastMonitoringService::postEvent);

    reg.watchPreSourceEvent(this,&FastMonitoringService::preSourceEvent);//source (with streamID of requestor)
    reg.watchPostSourceEvent(this,&FastMonitoringService::postSourceEvent);
  
    reg.watchPreModuleEvent(this,&FastMonitoringService::preModuleEvent);//should be stream
    reg.watchPostModuleEvent(this,&FastMonitoringService::postModuleEvent);//


    for(unsigned int i = 0; i < (mCOUNT); i++)
      encModule_.updateReserved((void*)(reservedMicroStateNames+i));
    encPath_.update((void*)&nopath_);
    encModule_.completeReservedWithDummies();

    // The run dir should be set via the configuration
    // For now, just grab the latest run directory available

    // FIND RUN DIRECTORY
    boost::filesystem::path runDirectory(edm::Service<evf::EvFDaqDirector>()->findHighestRunDir());
    workingDirectory_ = runDirectory_ = runDirectory;
    workingDirectory_ /= "mon";

    bool foundMonDir = false;
    if ( boost::filesystem::is_directory(workingDirectory_))
    	foundMonDir=true;
    if (!foundMonDir) {
    	std::cout << "<MON> DIR NOT FOUND!" << std::endl;
        boost::filesystem::create_directories(workingDirectory_);
    }

    std::ostringstream fastFileName;

    fastFileName << fastName_ << "_pid" << std::setfill('0') << std::setw(5) << getpid() << ".fast";
    boost::filesystem::path fast = workingDirectory_;
    fast /= fastFileName.str();
    fastPath_ = fast.string();

    /*
     * initialize the fast monitor with:
     * vector of pointers to monitorable parameters
     * path to definition
     *
     */
    std::cout
			<< "FastMonitoringService: initializing FastMonitor with microstate def path: "
			<< microstateDefPath_ << " " << FastMonitoringThread::MCOUNT << " "
			<< encPath_.current_ + 1 << " " << encModule_.current_ + 1
			<< std::endl;
  }


  FastMonitoringService::~FastMonitoringService()
  {
  }



  std::string FastMonitoringService::makePathLegenda(){
    
    std::ostringstream ost;
    for(int i = 0;
	i < encPath_.current_;
	i++)
      ost<<i<<"="<<*((std::string *)(encPath_.decode(i)))<<" ";
    return ost.str();
  }

  std::string FastMonitoringService::makeModuleLegenda(){
    
    std::ostringstream ost;
    for(int i = 0;
	i < encModule_.current_;
	i++)
      {
	//	std::cout << "for i = " << i << std::endl;
	ost<<i<<"="<<((const edm::ModuleDescription *)(encModule_.decode(i)))->moduleLabel()<<" ";
      }
    return ost.str();
  }





  void FastMonitoringService::preallocate(edm::service::SystemBounds const & bounds)
  {
    //we can begin monitoring at this step

    nStreams_=bounds.maxNumberOfStreams();
    nthreads_=bounds.maxNumberOfThreads();

    macrostate_=FastMonitoringThread::sInit;

    for (unsigned int i=0;i<nStreams_;i++) {
       fmt_.m_data.streamLumi_.push_back(0);
       ministate_.push_back(&nopath_);
       microstate_.push_back(&reservedMicroStateNames[mInvalid]);
    }
 
    //TODO: we could do fastpath output even before seeing lumi
    lastGlobalLumi_=0;//this means no fast path before begingGlobalLumi (for now), 
    isGlobalLumiTransition_=true;
    lumiFromSource_=0;

    //startup monitoring
    fmt_.resetFastMonitor(microstateDefPath_);
    fmt_.m_data.registerVariables(fmt_.jsonMonitor_, nStreams_);
    fmt_.start(&FastMonitoringService::dowork,this);
  }

  void FastMonitoringService::jobFailure()
  {
    fmt_.m_data.macrostate_ = FastMonitoringThread::sError;
  }



  //new output module name is stream
  void FastMonitoringService::preModuleBeginJob(const edm::ModuleDescription& desc)
  {
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
    //std::cout << "path legenda*****************" << std::endl;
   // std::cout << makePathLegenda()   << std::endl;
    std::cout << "module legenda***************" << std::endl;
    std::cout << makeModuleLegenda() << std::endl;
    fmt_.m_data.macrostate_ = FastMonitoringThread::sJobReady;
  }

  void FastMonitoringService::postEndJob()
  {
    macrostate_ = FastMonitoringThread::sJobEnded;
    fmt_.stop();
  }


  void FastMonitoringService::prePathBeginRun(const std::string& pathName)
  {
    return ;
    //bonus track, now monitoring path execution too...
    // here we are forced to use string keys...
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>update path map with " << pathName << std::endl;
    encPath_.update((void*)&pathName);
  }

  void FastMonitoringService::postGlobalBeginRun(edm::GlobalContext const& gc)
  {
    macrostate_ = FastMonitoringThread::sRunning;
  }


  //start new lumi (can be before last global end lumi)
  void FastMonitoringService::preGlobalBeginLumi(edm::GlobalContext const& gc)//edm::LuminosityBlockID const& iID, edm::Timestamp const& iTime)
  {
	  std::cout << "FastMonitoringService: Pre-begin LUMI: " << gc.luminosityBlockID().luminosityBlock() << std::endl;

	  timeval lumiStartTime;
	  gettimeofday(&lumiStartTime, 0);
	  unsigned int newLumi = gc.luminosityBlockID().luminosityBlock();

	  fmt_.monlock_.lock();


	  lumiStartTime_[newLumi]=timeval;
	  while (!lastGlobalLumisClosed_.empty()) {
		  //wipe out old map entries as they aren't needed and slow down access
		  unsigned int oldLumi = lastGlobalLumisClosed_.pop();
		  lumiStartTime_.erase(oldLumi);
		  throughput_.erase(oldLumi);
		  avgLeadTime_.erase(oldLumi);
		  filesProcessedDuringLumi_.erase(oldLumi);
		  accuSize_.erase(oldLumi);
		  processedEventsPerLumi_.erase(oldLumi);
		  streamEolMap_.erase(oldLumi);
	  }
	  lastGlobaLumi_= newLumi;
	  isGlobalLumiTransition_=false;

	  //put a map for streams to report if they had EOL (else we have to update some variables)
	  std::vector<bool> streamEolVec_;
	  for (unsigned int i=0;i<nStreams_;i++) streamEolVec_.push_back(false);
	  streamEoLMap_[newLumi] = streamEolVec_;

	  fmt_.monlock_.unlock();
  }

  //global end lumi (no streams will process this further)
  void FastMonitoringService::preGlobalEndLumi(edm::GlobalContext const& gc)
  {
	  unsigned int lumi = gc.luminosityBlockID().luminosityBlock();
	  std::cout << "FastMonitoringService: LUMI: " << lumi << " ended! Writing JSON information..." << std::endl;
	  timeval lumiStopTime;
	  gettimeofday(&lumiStopTime, 0);

	  fmt_.monlock_.lock();
	  // Compute throughput
	  unsigned int secondsForLumi = lumiStopTime.tv_sec - lumiStartTime_[lumi].tv_sec;
	  unsigned long accuSize = accuSize_[lumi]==map::endl ? 0 : accuSize_[lumi];
	  double throughput = double(accuSize) / double(secondsForLumi) / double(1024*1024);
	  //store to registered variable
	  fmt_.m_data.throughputJ_.value() = throughput;

	  //update
	  doSnapshot(false,lumi,true);

	  // create file name for slow monitoring file
	  std::stringstream slowFileName;
	  slowFileName << slowName_ << "_ls" << std::setfill('0') << std::setw(4)
			<< lumi << "_pid" << std::setfill('0')
			<< std::setw(5) << getpid() << ".jsn";
	  boost::filesystem::path slow = workingDirectory_;
	  slow /= slowFileName.str();

	  //retrieve one result we need (todo: sanity check if it's found)
	  IntJ* lumiProcessedJptr = std::dynamic_cast<IntJ*>(fmt_.jsonMonitor->getMergedVarForLumi("Processed",lumi));
          assert(lumiProcessedJpr!=nullptr);
	  processedEventsPerLumi_[lumi] = lumiProcessedJptr->value();

	  std::cout
			<< ">>> >>> FastMonitoringService: processed event count for this lumi = "
			<< fmt_.m_data.lumiProcessedJptr_->value() << " time = " << secondsForLumi
			<< " size = " << accuSize << " thr = " << throughput << std::endl;
	  delete lumiProcessedJptr;

	  //full global and stream merge&output for this lumi
	  fmt_.m_data.jsonMonitor_->outputFullJSON(slow.string(),lumi);//full global and stream merge and JSON write for this lumi
	  fmt_.m_data.jsonMonitor_->discardCollected(lumi);//we don't do further updates for this lumi

	  isGlobalLumiTransition_=true;
	  fmt_.monlock_.unlock();
  }

  void FastMonitoringService::postGlobalEndLumi(edm::GlobalContext const& gc)
  {
    //mark closed lumis (still keep map entries until next one)
    lastGlobalLumisClosed_.push(gc.luminosityBlockID().luminosityBlock());
  }

  void FastMonitoringService::preStreamBeginLumi(edm::Streamcontext const& sc)
  {
    std::cout << "FastMonitoringService: Pre-stream-begin LUMI: " << iID.luminosityBlock() << std::endl;
    unsigned int sid = sc.streamID().value();
    fmt_.monlock_.lock();
    fmt_.m_data.streamLumi_[sid] = sc.luminosityBlockID().luminosityBlock();

    //reset collected values for this stream
    fmt_.m_data.processed_[sid]=0;
    ministate_[sid]=&nopath_;
    microstate_[sid]=reservedMicroStateNames[mInvalid];
    fmt_.monlock_.unlock();
  }

  void FastMonitoringService::preStreamEndLumi(edm::Streamcontext const& sc)
  {
    std::cout << "FastMonitoringService: Pre-stream-end LUMI: " << iID.luminosityBlock() << std::endl;
    unsigned int sid = sc.streamID().value();
    fmt_.monlock_.lock();
    //update processed count to be complete at this time (event if it's still snapped before global EOL or not)
    doStreamEOLSnapshot(false,forLumi,sid);
    //reset this in case stream does not get notified of next lumi (we keep processed events only)
    ministate_[sid]=&nopath_;
    microstate_[sid]=reservedMicroStateNames[mInvalid];
    fmt_.monlock_.unlock();
  }


  void FastMonitoringService::prePathEvent(edm::StreamContext const& sc, const edm::PathContext const& pc)
  {
    //use relaxed, as we also check using streamID and eventID
    if (!collectedPathList_.load(std::memory_order_relaxed))
    {
      if (sc.streamID().value()!=0) return;
      initPathsLock_.lock();
      if (firstEventId_==0) 
	firstEventId_==sc.eventID().value();
      if (sc.eventID().value()==firstEventId_)
      {
	encPath_.update((void*)&pc.pathName());
	initPathsLock_.unlock();
	return;
      }
      else {
	collectedPathList_.store(true,std::memory_order_seq_cst);
	initPathsLock_.unlock();
	//print paths
	//finished collecting path names
	std::cout << "path legenda*****************" << std::endl;
	std::cout << makePathLegenda()   << std::endl;
      }
    }
    ministate_ = &(pc.pathName());
  }


  void FastMonitoringService::preEvent(edm::StreamContext const& sc)
  {
  }

  //shoudl be synchronized before
  void FastMonitoringService::postEvent(edm::StreamContext const& sc)
  {
    //it is possible that mutex is needed for synchronization with endOfLumi depending what CMSSW does between tasks
    //(unless it does full memory fencing between tasks)

    fmt_.m_data.microstate_ = &reservedMicroStateNames[mFwkOvh];
    fmt_.m_data.ministate_ = &nopath;
    //fmt_.monlock_.lock();
//    fmt_.m_data.processed_[sc.streamID()].fetch_add(1,std::memory_order_release);
    fmt_.m_data.processed_[sc.streamID()].fetch_add(1,std::memory_order_relaxed);//no ordering required
    //fmt_.monlock_.unlock();
  }

  void FastMonitoringService::preSourceEvent(edm::StreamID sid)
  {
    microstate_[sid.value()] = &reservedMicroStateNames[mIdle];
  }

  void FastMonitoringService::postSourceEvent(edm::StreamID sid)
  {
    microstate_[sid.value()] = &reservedMicroStateNames[mFwkOvh];
  }

  void FastMonitoringService::preModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc)
  {
    microstate_[sc.streamID().value()] = &(mcc.moduleDescription());
  }

  void FastMonitoringService::postModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc)
  {
    microstate_[sc.streamID().value()] = &(mcc.moduleDescription());
  }

  //FUNCTIONS CALLED FROM OUTSIDE

  //this is for old-fashioned service that is not thread safe and can block other streams
  //(we assume the worst case - everything is blocked)
  void FastMonitoringService::setMicroState(Microstate m)
  {
    for (int i=0;i<nStreams_;i++)
    microstate_[i] = &reservedMicroStateNames[m];
  }

  //this is for another service that is thread safe or rarely blocks other streams
  void FastMonitoringService::setMicroState(edm::StreamID sid, Microstate m)
  {
    microstate_[sid] = &reservedMicroStateNames[m];
  }

  //from source - needs changes to source to track per-lumi file processing
  void FastMonitoringService::accumulateFileSize(unsigned int lumi, unsigned long fileSize) {
	fmt_.monlock_.lock();
	//std::cout << "--> ACCUMMULATING size: " << fileSize << std::endl;
	if (accuSize_.find(lumi)==std::map::end) accuSize_[lumi] = fileSize;
	else accuSize_[lumi] += fileSize;

	if (filesProcessedDuringLumi_.find(lumi)==map::end)
	  filesProcessedDuringLumi_[lumi] = fileSize;
	else
	  fmt_.m_data.filesProcessedDuringLumi_[lumi]++;
	fmt_.monlock_.unlock();
  }

  //this seems to assign the elapsed time to previous lumi if there is a boundary
  //can be improved, especially with threaded input reading
  void FastMonitoringService::startedLookingForFile() {//this should be measured in source, probably (TODO)
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
	  fmt_.monlock_.lock();
	  if (lumi>lumiFromSource) {
		  //source got files for new lumi
		  lumiFromSource_=lumi;
		  leadTimes_.clear();
	  }
	  double elapsedTime = (fileLookStop_.tv_sec - fileLookStart_.tv_sec) * 1000.0; // sec to ms
	  elapsedTime += (fileLookStop_.tv_usec - fileLookStart_.tv_usec) / 1000.0; // us to ms
	  // add this to lead times for this lumi
	  leadTimes_.push_back(elapsedTime);

	  // recompute average lead time for this lumi
	  if (leadTimes_.size() == 1) avgLeadTime_[lumi] = leadTimes_[0];
	  else {
		  double totTime = 0;
		  for (unsigned int i = 0; i < leadTimes_.size(); i++) totTime += leadTimes_[i];
		  avgLeadTime_[lumi] = totTime / leadTimes_.size();
	  }
	  fmt_.monlock_.unlock();
  }

  //needs multithreading-aware output module
  unsigned int FastMonitoringService::getEventsProcessedForLumi(unsigned int lumi) {
	fmt_.monlock_.lock();
	auto it = processedEventsPerLumi_.find(lumi);
	if (it!=std::map::end) {
	  unsigned int proc = *it;
	  fmt_.monlock_.unlock();
	  return proc;
	}
	else {
	        fmt_.monlock_.unlock();
		std::cout << "ERROR: output module wants deleted info " << std::endl;
		return 0;
}

} //end namespace evf

