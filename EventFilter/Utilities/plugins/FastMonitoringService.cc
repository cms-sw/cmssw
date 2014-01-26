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
 
    //TODO: we could do fastpath of output, inputSource vars, processed=0 and m*states, i.e. start from lumi=1;
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
    //    boost::mutex::scoped_lock sl(lock_);
    std::cout << "path legenda*****************" << std::endl;
    std::cout << makePathLegenda()   << std::endl;
    std::cout << "module legenda***************" << std::endl;
    std::cout << makeModuleLegenda() << std::endl;
    fmt_.m_data.macrostate_ = FastMonitoringThread::sJobReady;
  }

  void FastMonitoringService::postEndJob()
  {
    //    boost::mutex::scoped_lock sl(lock_);
    macrostate_ = FastMonitoringThread::sJobEnded;
    fmt_.stop();
  }


  //TODO:find alternative way to get path names
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
    return;
    std::cout << "path legenda*****************" << std::endl;
//    std::cout << makePathLegenda()   << std::endl; //TODO
    macrostate_ = FastMonitoringThread::sRunning;
  }


  //start new lumi (can be before last global end lumi)
  void FastMonitoringService::preGlobalBeginLumi(edm::GlobalContext const& gc)//edm::LuminosityBlockID const& iID, edm::Timestamp const& iTime)
  {
	  std::cout << "FastMonitoringService: Pre-begin LUMI: " << iID.luminosityBlock() << std::endl;

	  fmt_.monlock_.lock();

	  timeval lumiStartTime;
	  gettimeofday(&lumiStartTime, 0);
	  unsigned int newLumi = gc.luminosityBlockID().luminosityBlock();

	  lumiStartTime_[newLumi]=timeval;
	  if (lastGlobalLumi_>0) {
		  //wipe out old map entries as they aren't needed (TODO)
	  }
	  lastGlobaLumi_= newLumi;
	  isGlobalLumiTransition_=false;
	  fmt_.monlock_.unlock();
  }

  //global end lumi
  void FastMonitoringService::preGlobalEndLumi(edm::GlobalContext const& gc)//edm::LuminosityBlockID const& iID, edm::Timestamp const& iTime)
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

	  std::cout
			<< ">>> >>> FastMonitoringService: processed event count for this lumi = "
			<< fmt_.m_data.processedJ_.value() << " time = " << secondsForLumi
			<< " size = " << accuSize << " thr = " << throughput << std::endl;

	  doSnapshot(true,lumi);

	  // create file name for slow monitoring file
	  std::stringstream slowFileName;
	  slowFileName << slowName_ << "_ls" << std::setfill('0') << std::setw(4)
			<< lumi << "_pid" << std::setfill('0')
			<< std::setw(5) << getpid() << ".jsn";
	  boost::filesystem::path slow = workingDirectory_;
	  slow /= slowFileName.str();

	  //retrieve one result we need (todo check if it's found)
	  intJ lumiProcessedJ = fmt_.m_data.jsonMonitor->getMergedIntJforLumi("Processed",lumi);//TODO
	  processedEventsPerLumi_[lumi] = lumiProcessedJ.value();//fmt_.m_data.processedJ_.value();

	  //full global and stream merge&output for this lumi
	  fmt_.m_data.jsonMonitor_->outputFullHistoDataPoint(slow.string());//full global and stream merge&output for this lumi
	  fmt_.m_data.jsonMonitor_->discardCollected(lumi);//TODO:see what we do between next beginLumi

	  //cleanup of global per lumi info
	  fmt_.m_data.avgLeadTime_.erase(lumi);
	  fmt_.m_data.filesProcessedDuringLumi_.erase(lumi);

	  //not directly monitored:
	  accuSize_.erase(lumi);
	  globallumi_.erase(lumi);//safe ?

	  isGlobalLumiTransition_=true;
	  fmt_.monlock_.unlock();
  }

  //TODO add another concurrent queue here
  void FastMonitoringService::preStreamBeginLumi(edm::Streamcontext const& sc)
  {
    std::cout << "FastMonitoringService: Pre-begin LUMI: " << iID.luminosityBlock() << std::endl;
    unsigned int sid = sc.streamID().value();
    fmt_.monlock_.lock();
    fmt_.m_data.streamLumi_[sid] = sc.luminosityBlockID().luminosityBlock();

    //reset collected values for this stream
    fmt_.m_data.processed_[sid]=0;
    ministate_[sid]=&nopath_;
    microstate_[sid]=reservedMicroStateNames[mInvalid];
    fmt_.monlock_.unlock();
  }

  //todo:where does input belong here? queue next?
  void FastMonitoringService::preStreamEndLumi(edm::Streamcontext const& gc)
  {
    std::cout << "FastMonitoringService: Pre-begin LUMI: " << iID.luminosityBlock() << std::endl;
    fmt_.monlock_.lock();
    //no update needed
    //streamlumi_[sc.streamID().value()] = sc.luminosityBlockID().luminosityBlock();
    fmt_.monlock_.unlock();
  }


  void FastMonitoringService::prePathEvent(edm::StreamContext const& sc, const edm::PathContext const& pc)
  {
/* do nothing for now
    //bonus track, now monitoring path execution too...
    if (firstEvent) {
	    sc.eventID().event()
    }
    ministate_ = &(pc.pathName());
    */
  }


  void FastMonitoringService::preEvent(edm::StreamContext const& sc)
  {
  }

  void FastMonitoringService::postEvent(edm::StreamContext const& sc)
  {
    fmt_.m_data.microstate_ = &reservedMicroStateNames[mFwkOvh];
    fmt_.monlock_.lock();
    fmt_.m_data.processed_[sc.streamID()].value()++;
    fmt_.monlock_.unlock();
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
	if (processedEventsPerLumi_[lumi]!=std::map::endl) {
	  return processedEventsPerLumi_[lumi];//should delete this entry in postGlobalEndLumi?
	}
	else {
		std::cout << "output module wants deleted info " << std::endl;/TODO:info message
		return 0;
}

} //end namespace evf

