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
  	,sleepTime_(iPS.getUntrackedParameter<int>("sleepTime", 1))
    //,rootDirectory_(iPS.getUntrackedParameter<string>("rootDirectory", "/data"))
    ,microstateDefPath_(iPS.getUntrackedParameter<string>("microstateDefPath", "/tmp/def.jsd"))
    ,outputDefPath_(iPS.getUntrackedParameter<string>("outputDefPath", "/tmp/def.jsd"))
    ,fastName_(iPS.getUntrackedParameter<string>("fastName", "states"))
    ,slowName_(iPS.getUntrackedParameter<string>("slowName", "lumi"))
  {
    fmt_.m_data.macrostate_=FastMonitoringThread::sInit;
    fmt_.m_data.ministate_=&nopath_;
    fmt_.m_data.microstate_=&reservedMicroStateNames[mInvalid];
    fmt_.m_data.lumisection_ = 0;
    fmt_.m_data.accuSize_ = 0;
    fmt_.m_data.filesProcessedDuringLumi_ = 0;
    reg.watchPreModuleBeginJob(this,&FastMonitoringService::preModuleBeginJob);  
    reg.watchPreBeginLumi(this,&FastMonitoringService::preBeginLumi);
    reg.watchPreEndLumi(this,&FastMonitoringService::preEndLumi);
    reg.watchPrePathBeginRun(this,&FastMonitoringService::prePathBeginRun);
    reg.watchPostBeginJob(this,&FastMonitoringService::postBeginJob);
    reg.watchPostBeginRun(this,&FastMonitoringService::postBeginRun);
    reg.watchPostEndJob(this,&FastMonitoringService::postEndJob);
    reg.watchPreProcessPath(this,&FastMonitoringService::preProcessPath);
    reg.watchPreProcessEvent(this,&FastMonitoringService::preEventProcessing);
    reg.watchPostProcessEvent(this,&FastMonitoringService::postEventProcessing);
    reg.watchPreSource(this,&FastMonitoringService::preSource);
    reg.watchPostSource(this,&FastMonitoringService::postSource);
  
    reg.watchPreModule(this,&FastMonitoringService::preModule);
    reg.watchPostModule(this,&FastMonitoringService::postModule);
    reg.watchJobFailure(this,&FastMonitoringService::jobFailure);
    for(unsigned int i = 0; i < (mCOUNT); i++)
      encModule_.updateReserved((void*)(reservedMicroStateNames+i));
    encPath_.update((void*)&nopath_);
    encModule_.completeReservedWithDummies();

    fmt_.m_data.macrostateJ_.setName("Macrostate");
    fmt_.m_data.ministateJ_.setName("Ministate");
    fmt_.m_data.microstateJ_.setName("Microstate");
    fmt_.m_data.processedJ_.setName("Processed");
    fmt_.m_data.throughputJ_.setName("Throughput");
    fmt_.m_data.avgLeadTimeJ_.setName("AverageLeadTime");
    fmt_.m_data.filesProcessedDuringLumi_.setName("FilesProcessed");
    vector<JsonMonitorable*> monParams;
    monParams.push_back(&fmt_.m_data.macrostateJ_);
    monParams.push_back(&fmt_.m_data.ministateJ_);
    monParams.push_back(&fmt_.m_data.microstateJ_);
    monParams.push_back(&fmt_.m_data.processedJ_);
    monParams.push_back(&fmt_.m_data.throughputJ_);
    monParams.push_back(&fmt_.m_data.avgLeadTimeJ_);
    monParams.push_back(&fmt_.m_data.filesProcessedDuringLumi_);

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

    fmt_.m_data.jsonMonitor_.reset(
			new FastMonitor(monParams, microstateDefPath_));

    fmt_.start(&FastMonitoringService::dowork,this);
  }


  FastMonitoringService::~FastMonitoringService()
  {
  }

  void FastMonitoringService::preModuleBeginJob(const edm::ModuleDescription& desc)
  {
    //build a map of modules keyed by their module description address
    //here we need to treat output modules in a special way so they can be easily singled out
    if(desc.moduleName() == "ShmStreamConsumer" || desc.moduleName() == "EventStreamFileWriter" || 
       desc.moduleName() == "PoolOutputModule")
      encModule_.updateReserved((void*)&desc);
    else
      encModule_.update((void*)&desc);
  }

  void FastMonitoringService::prePathBeginRun(const std::string& pathName)
  {
    //bonus track, now monitoring path execution too...
    // here we are forced to use string keys...
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

  void FastMonitoringService::preBeginLumi(edm::LuminosityBlockID const& iID, edm::Timestamp const& iTime)
  {
	  std::cout << "FastMonitoringService: Pre-begin LUMI: " << iID.luminosityBlock() << std::endl;
	  fmt_.monlock_.lock();
	  fmt_.m_data.lumisection_ = (unsigned int) iID.luminosityBlock();
	  gettimeofday(&lumiStartTime_, 0);
	  fmt_.monlock_.unlock();
  }

  void FastMonitoringService::preEndLumi(edm::LuminosityBlockID const& iID, edm::Timestamp const& iTime)
  {
	  std::cout << "FastMonitoringService: LUMI: " << iID.luminosityBlock() << " ended! Writing JSON information..." << std::endl;
	  fmt_.monlock_.lock();
	  gettimeofday(&lumiStopTime_, 0);

	  // Compute throughput
	  unsigned int secondsForLumi = lumiStopTime_.tv_sec - lumiStartTime_.tv_sec;
	  fmt_.m_data.throughputJ_.value() = double(fmt_.m_data.accuSize_) / double(secondsForLumi) / double(1024*1024);

	  std::cout
			<< ">>> >>> FastMonitoringService: processed event count for this lumi = "
			<< fmt_.m_data.processedJ_.value() << " time = " << secondsForLumi
			<< " size = " << fmt_.m_data.accuSize_ << " thr = " << fmt_.m_data.throughputJ_.value() << std::endl;
	  fmt_.m_data.jsonMonitor_->snap(true, fastPath_);
	  // create file name for slow monitoring file
	  std::stringstream slowFileName;
	  slowFileName << slowName_ << "_ls" << std::setfill('0') << std::setw(4)
			<< fmt_.m_data.lumisection_ << "_pid" << std::setfill('0')
			<< std::setw(5) << getpid() << ".jsn";
	  boost::filesystem::path slow = workingDirectory_;
	  slow /= slowFileName.str();
	  fmt_.m_data.jsonMonitor_->outputFullHistoDataPoint(slow.string());
	  processedEventsPerLumi_[fmt_.m_data.lumisection_] = fmt_.m_data.processedJ_.value();

	  fmt_.m_data.processedJ_ = 0;
	  fmt_.m_data.accuSize_ = 0;
	  fmt_.m_data.throughputJ_ = 0;
	  fmt_.m_data.avgLeadTimeJ_ = 0;
	  fmt_.m_data.filesProcessedDuringLumi_ = 0;
	  leadTimes_.clear();
	  fmt_.monlock_.unlock();
  }

  void FastMonitoringService::preEventProcessing(const edm::EventID& iID,
					     const edm::Timestamp& iTime)
  {
    //    boost::mutex::scoped_lock sl(lock_);
  }

  void FastMonitoringService::postEventProcessing(const edm::Event& e, const edm::EventSetup&)
  {
    //    boost::mutex::scoped_lock sl(lock_);
    fmt_.m_data.microstate_ = &reservedMicroStateNames[mFwkOvh];
    fmt_.monlock_.lock();
    fmt_.m_data.processedJ_.value()++;
    fmt_.monlock_.unlock();
  }
  void FastMonitoringService::preSource()
  {
    //    boost::mutex::scoped_lock sl(lock_);
    fmt_.m_data.microstate_ = &reservedMicroStateNames[mIdle];
  }

  void FastMonitoringService::postSource()
  {
    //    boost::mutex::scoped_lock sl(lock_);
    fmt_.m_data.microstate_ = &reservedMicroStateNames[mFwkOvh];
  }

  void FastMonitoringService::preModule(const edm::ModuleDescription& desc)
  {
    //    boost::mutex::scoped_lock sl(lock_);
    fmt_.m_data.microstate_ = &desc;
  }

  void FastMonitoringService::postModule(const edm::ModuleDescription& desc)
  {
    //    boost::mutex::scoped_lock sl(lock_);
    fmt_.m_data.microstate_ = &desc;
  }
  void FastMonitoringService::jobFailure()
  {
    //    boost::mutex::scoped_lock sl(lock_);
    fmt_.m_data.macrostate_ = FastMonitoringThread::sError;
  }
  void FastMonitoringService::setMicroState(Microstate m)
  {
    //    boost::mutex::scoped_lock sl(lock_);
    fmt_.m_data.microstate_ = &reservedMicroStateNames[m];
  }

  void FastMonitoringService::accummulateFileSize(unsigned long fileSize) {
	fmt_.monlock_.lock();
	//std::cout << "--> ACCUMMULATING size: " << fileSize << std::endl;
	fmt_.m_data.accuSize_ += fileSize;
	fmt_.m_data.filesProcessedDuringLumi_.value()++;
	fmt_.monlock_.unlock();
  }

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

  unsigned int FastMonitoringService::getEventsProcessedForLumi(unsigned int lumi) {
	return processedEventsPerLumi_[lumi];
}

} //end namespace evf

