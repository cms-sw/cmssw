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

    reg.watchPreModuleBeginJob(this,&FastMonitoringService::preModuleBeginJob);//global

    reg.watchPreGlobalBeginLumi(this,&FastMonitoringService::preBeginLumi);//global lumi
    reg.watchPreGlobalEndLumi(this,&FastMonitoringService::preEndLumi);

    reg.watchPreStreamBeginLumi(this,&FastMonitoringService::preBeginLumi);//stream lumi
    reg.watchPreStreamEndLumi(this,&FastMonitoringService::preEndLumi);

    reg.watchPrePathBeginRun(this,&FastMonitoringService::prePathBeginRun);//there is no equivalent of this (will need to get it at first event..)
    reg.watchPostBeginJob(this,&FastMonitoringService::postBeginJob);
    reg.watchPostBeginRun(this,&FastMonitoringService::postBeginRun);
    reg.watchPostEndJob(this,&FastMonitoringService::postEndJob);
    reg.watchPreProcessPath(this,&FastMonitoringService::preProcessPath);
    reg.watchPreEvent(this,&FastMonitoringService::preEventProcessing);//stream
    reg.watchPostEvent(this,&FastMonitoringService::postEventProcessing);//stream

    reg.watchPreSourceEvent(this,&FastMonitoringService::preSourceEvent);//source (with streamID of requestor)
    reg.watchPostSourceEvent(this,&FastMonitoringService::postSourceEvent);//source (with streamID )
  
    reg.watchPreModuleEvent(this,&FastMonitoringService::preModule);//should be stream
    reg.watchPostModuleEvent(this,&FastMonitoringService::postModule);//

    reg.watchJobFailure(this,&FastMonitoringService::jobFailure);//global

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

    fmt_.resetFastMonitor(microstateDefPath_);

    fmt_.start(&FastMonitoringService::dowork,this);
  }


  FastMonitoringService::~FastMonitoringService()
  {
  }

  void FastMonitoringService::preallocate(edm::service::SystemBounds const & bounds)
  {
    nStreams_=bounds.maxNumberOfStreams();
    nthreads_=bounds.maxNumberOfThreads();

    for (unsigned int i=0;i<nStreams_;i++) {
       fmt_.m_data.ministate_.push_back(&nopath_);
       fmt_.m_data.microstate_.push_back(&reservedMicroStateNames[mInvalid]);
       fmt_.m_data.streamlumi.push_back(0);
    }
    fmt_.m_data.processed_=0;
    //fmt_.m_data.macrostate_=FastMonitoringThread::sInit;
    //fmt_.m_data.lumisection_ = 0;
    //fmt_.m_data.accuSize_ = 0;
    //fmt_.m_data.filesProcessedDuringLumi_ = 0;
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

  void FastMonitoringService::prePathBeginRun(const std::string& pathName)
  {
    //bonus track, now monitoring path execution too...
    // here we are forced to use string keys...
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>update path map with " << pathName << std::endl;
    encPath_.update((void*)&pathName);
  }

  void FastMonitoringService::postBeginRun(edm::Run const&, edm::EventSetup const&)
  {
    std::cout << "path legenda*****************" << std::endl;
    std::cout << makePathLegenda()   << std::endl;
    fmt_.m_data.macrostate_ = FastMonitoringThread::sRunning;
  }

  void FastMonitoringService::preProcessPath(const std::string& pathName)
  {
    //bonus track, now monitoring path execution too...
    fmt_.m_data.ministate_ = &pathName;
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
    fmt_.m_data.macrostate_ = FastMonitoringThread::sJobEnded;
    fmt_.stop();
  }

  //start new lumi (can be before last global end lumi)
  void FastMonitoringService::preGlobalBeginLumi(edm::GlobalContext const& gc)//edm::LuminosityBlockID const& iID, edm::Timestamp const& iTime)
  {
	  std::cout << "FastMonitoringService: Pre-begin LUMI: " << iID.luminosityBlock() << std::endl;
	  timeval lumiStartTime_;
	  gettimeofday(&lumiStartTime, 0);

	  fmt_.monlock_.lock();
	  //insert
	  lumiStartTime_[gc.luminosityBlockID().luminosityBlock()]=timeval;
	  fmt_.m_data.lumisection_[gc.luminosityBlockID().luminosityBlock()] = true;//needed?
	  fmt_.monlock_.unlock();
  }

  //global end lumi
  void FastMonitoringService::preEndLumi(edm::GlobalContext const& gc)//edm::LuminosityBlockID const& iID, edm::Timestamp const& iTime)
  {
	  unsigned int lumi = gc.luminosityBlockID().luminosityBlock();
	  std::cout << "FastMonitoringService: LUMI: " << lumi << " ended! Writing JSON information..." << std::endl;
	  timeval lumiStopTime;
	  gettimeofday(&lumiStopTime, 0);

	  fmt_.monlock_.lock();
	  // Compute throughput
	  unsigned int secondsForLumi = lumiStopTime_.tv_sec - lumiStartTime_[lumi].tv_sec;
	  unsigned long accuSize = accuSize_[lumi]==map::endl ? 0 : accuSize_[lumi];
	  double throughput = double(accuSize) / double(secondsForLumi) / double(1024*1024);
	  //store to registered variable
	  fmt_.m_data.throughputJ_.value() = throughput;

	  std::cout
			<< ">>> >>> FastMonitoringService: processed event count for this lumi = "
			<< fmt_.m_data.processedJ_.value() << " time = " << secondsForLumi
			<< " size = " << accuSize << " thr = " << throughput << std::endl;

	  fmt_.m_data.jsonMonitor_->snap(true, fastPath_);//do a global snap here (TODO)

	  // create file name for slow monitoring file
	  std::stringstream slowFileName;
	  slowFileName << slowName_ << "_ls" << std::setfill('0') << std::setw(4)
			<< lumi << "_pid" << std::setfill('0')
			<< std::setw(5) << getpid() << ".jsn";
	  boost::filesystem::path slow = workingDirectory_;
	  slow /= slowFileName.str();

	  //retrieve one result we need (todo check if it's found)
	  intJ lumiProcessedJ = fmt_.m_data.jsonMonitor->getMergedIntJforLumi("name",lumi);//TODO
	  processedEventsPerLumi_[lumi] = lumiProcessedJ.value();//fmt_.m_data.processedJ_.value();

	  //full global and stream merge&output for this lumi
	  fmt_.m_data.jsonMonitor_->outputFullHistoDataPoint(slow.string());//full global and stream merge&output for this lumi
	  fmt_.m_data.jsonMonitor_->discardCollected(lumi);//TODO:see what we do between next beginLumi


	  //cleanup of global per lumi info
	  fmt_.m_data.avgLeadTimeJ_.erase(lumi);
	  fmt_.m_data.filesProcessedDuringLumi_.erase(lumi);

	  //not directly monitored:
	  accuSize_.erase(lumi);
	  leadTimes_.clear();
	  fmt_.m_data.lumisection_.erase(lumi);//safe ?

	  fmt_.monlock_.unlock();
  }

  //TODO add another concurrent queue here
  void FastMonitoringService::preStreamBeginLumi(edm::Streamcontext const& gc)//edm::LuminosityBlockID const& iID, edm::Timestamp const& iTime)
  {
	  std::cout << "FastMonitoringService: Pre-begin LUMI: " << iID.luminosityBlock() << std::endl;
	  fmt_.monlock_.lock();
	  fmt_.m_data.streamlumi_[sc.streamID().value()] = sc.luminosityBlockID().luminosityBlock();
	  //gettimeofday(&streamlumiStartTime_, 0);
	  fmt_.monlock_.unlock();
  }

  //todo:where does input belong here? queue next?
  void FastMonitoringService::preStreamEndLumi(edm::Streamcontext const& gc)//edm::LuminosityBlockID const& iID, edm::Timestamp const& iTime)
  {
	  std::cout << "FastMonitoringService: Pre-begin LUMI: " << iID.luminosityBlock() << std::endl;
	  fmt_.monlock_.lock();
	  fmt_.m_data.streamlumi_[sc.streamID().value()] = sc.luminosityBlockID().luminosityBlock();
	  //gettimeofday(&streamlumiStartTime_, 0);
	  fmt_.monlock_.unlock();
  }



  void FastMonitoringService::preEventProcessing(StreamContext const&) //const edm::EventID& iID,const edm::Timestamp& iTime)
  {
    //    boost::mutex::scoped_lock sl(lock_);
  }

  //@SM: this is OK
  void FastMonitoringService::postEventProcessing(StreamContext const&)//const edm::Event& e, const edm::EventSetup&)
  {
    //    boost::mutex::scoped_lock sl(lock_);
    fmt_.m_data.microstate_ = &reservedMicroStateNames[mFwkOvh];
    fmt_.monlock_.lock();
    fmt_.m_data.processedJ_.value()++;
    fmt_.monlock_.unlock();
  }

  void FastMonitoringService::preSourceEvent(edm::StreamID sid)
  {
    fmt_.m_data.microstate_[sid.value()] = &reservedMicroStateNames[mIdle];
  }

  void FastMonitoringService::postSourceEvent(edm::StreamID sid)
  {
    fmt_.m_data.microstate_[sid.value()] = &reservedMicroStateNames[mFwkOvh];
  }

  void FastMonitoringService::preModule(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) //const edm::ModuleDescription& desc)
  {
    fmt_.m_data.microstate_[sc.streamID().value()] = &(mcc.moduleDescription());
  }

  void FastMonitoringService::postModule(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) //const edm::ModuleDescription& desc)
  {
    fmt_.m_data.microstate_[sc.streamID().value()] = &(mcc.moduleDescription());
  }

  void FastMonitoringService::jobFailure()
  {
    fmt_.m_data.macrostate_ = FastMonitoringThread::sError;
  }

  //this is for old-fashioned service that is not thread safe and can block other streams
  //(we assume the worst case - everything is blocked)
  void FastMonitoringService::setMicroState(Microstate m)
  {
    for (int i=0;i<nStreams_;i++)
    fmt_.m_data.microstate_[i] = &reservedMicroStateNames[m];
  }

  //this is for another service that is thread safe or rarely blocks other streams
  void FastMonitoringService::setMicroState(edm::StreamID sid, Microstate m)
  {
    fmt_.m_data.microstate_[i] = &reservedMicroStateNames[m];
  }

  //from source - needs changes to source to track per-lumi file processing
  void FastMonitoringService::accumulateFileSize(unsigned int lumi, unsigned long fileSize) {
	fmt_.monlock_.lock();
	//std::cout << "--> ACCUMMULATING size: " << fileSize << std::endl;
	if (accuSize_[lumi]==map::endl) accuSize_[lumi] = fileSize;
	else accuSize_[lumi] += fileSize;

	if (fmt_.m_data.filesProcessedDuringLumi_[lumi]==map::endl)
	  mt_.m_data.filesProcessedDuringLumi_[lumi].value() = fileSize;
	else
	  fmt_.m_data.filesProcessedDuringLumi_[lumi].value()++;
	fmt_.monlock_.unlock();
  }

  //this seems to assign the elapsed time to previous lumi if there is a boundary
  //can be improved, especially with threaded input reading
  void FastMonitoringService::startedLookingForFile() {
	  gettimeofday(&fileLookStart_, 0);
	  /*
	 std::cout << "Started looking for .raw file at: s=" << fileLookStart_.tv_sec << ": ms = "
	 << fileLookStart_.tv_usec / 1000.0 << std::endl;
	 */
  }

  void FastMonitoringService::stoppedLookingForFile() {
	  gettimeofday(&fileLookStop_, 0);
	  /*
	 std::cout << "Stopped looking for .raw file at: s=" << fileLookStop_.tv_sec << ": ms = "
	 << fileLookStop_.tv_usec / 1000.0 << std::endl;
	 */
	  double elapsedTime = (fileLookStop_.tv_sec - fileLookStart_.tv_sec) * 1000.0; // sec to ms
	  elapsedTime += (fileLookStop_.tv_usec - fileLookStart_.tv_usec) / 1000.0; // us to ms
	  // add this to lead times for this lumi
	  leadTimes_.push_back(elapsedTime);

	  // recompute average lead time for this lumi
	  if (leadTimes_.size() == 1) fmt_.m_data.avgLeadTimeJ_ = leadTimes_[0];
	  else {
		  double totTime = 0;
		  for (unsigned int i = 0; i < leadTimes_.size(); i++) totTime += leadTimes_[i];
		  fmt_.m_data.avgLeadTimeJ_ = totTime / leadTimes_.size();
	  }
  }

  //needs multithreading-aware output module
  unsigned int FastMonitoringService::getEventsProcessedForLumi(unsigned int lumi) {
	return processedEventsPerLumi_[lumi];//should delete this entry in postGlobalEndLumi
}

} //end namespace evf

