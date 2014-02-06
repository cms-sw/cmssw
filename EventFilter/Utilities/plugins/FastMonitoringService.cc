#include "EventFilter/Utilities/plugins/FastMonitoringService.h"
#include <iostream>

#include "FWCore/Framework/interface/Event.h"
#include <iomanip>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "EventFilter/Utilities/plugins/EvFDaqDirector.h"


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
    //,rootDirectory_(iPS.getUntrackedParameter<std::string>("rootDirectory", "/data"))
    ,microstateDefPath_(iPS.getUntrackedParameter<std::string>("microstateDefPath", "/tmp/def.jsd"))
    ,outputDefPath_(iPS.getUntrackedParameter<std::string>("outputDefPath", "/tmp/def.jsd"))
    ,fastName_(iPS.getUntrackedParameter<std::string>("fastName", "states"))
    ,slowName_(iPS.getUntrackedParameter<std::string>("slowName", "lumi"))
  {
    reg.watchPreallocate(this, &FastMonitoringService::preallocate);//receiving information on number of threads
    reg.watchJobFailure(this,&FastMonitoringService::jobFailure);//global

    reg.watchPreModuleBeginJob(this,&FastMonitoringService::preModuleBeginJob);//global
    reg.watchPostBeginJob(this,&FastMonitoringService::postBeginJob);
    reg.watchPostEndJob(this,&FastMonitoringService::postEndJob);

    //reg.watchPrePathBeginRun(this,&FastMonitoringService::prePathBeginRun);//there is no equivalent of this now

    reg.watchPreGlobalBeginLumi(this,&FastMonitoringService::preGlobalBeginLumi);//global lumi
    reg.watchPreGlobalEndLumi(this,&FastMonitoringService::preGlobalEndLumi);
    reg.watchPostGlobalEndLumi(this,&FastMonitoringService::postGlobalEndLumi);

    reg.watchPreStreamBeginLumi(this,&FastMonitoringService::preStreamBeginLumi);//stream lumi
    reg.watchPreStreamEndLumi(this,&FastMonitoringService::preStreamEndLumi);

    reg.watchPrePathEvent(this,&FastMonitoringService::prePathEvent);

    reg.watchPreEvent(this,&FastMonitoringService::preEvent);//stream
    reg.watchPostEvent(this,&FastMonitoringService::postEvent);

    reg.watchPreSourceEvent(this,&FastMonitoringService::preSourceEvent);//source (with streamID of requestor)
    reg.watchPostSourceEvent(this,&FastMonitoringService::postSourceEvent);
  
    reg.watchPreModuleEvent(this,&FastMonitoringService::preModuleEvent);//should be stream
    reg.watchPostModuleEvent(this,&FastMonitoringService::postModuleEvent);//


    for(unsigned int i = 0; i < (mCOUNT); i++)
      encModule_.updateReserved((void*)(reservedMicroStateNames+i));
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
			//<< encPath_.current_ + 1 << " " << encModule_.current_ + 1
			<< std::endl;

   //#endif
   //#if TBB_IMPLEMENT_CPP0X
   ////std::cout << "TBB thread id:" <<  tbb::thread::id() << std::endl;
   //threadIDAvailable_=true;
   //#endif
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
    nThreads_=bounds.maxNumberOfThreads();

    //use same approach even if no multithreading
    if (nStreams_==0) nStreams_=1;
    if (nThreads_==0) nThreads_=1;
    //TODO: what if nThreads_<nStreams?
    macrostate_=FastMonitoringThread::sInit;

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
    //#if TBB_IMPLEMENT_CPP0X
    //for (unsigned int i=0;i<nThreads_;i++)
    //  threadMicrostate_.push_back(&reservedMicroStateNames[mInvalid]);
    //#endif

    //initial size until we detect number of bins
    fmt_.m_data.macrostateBins_=FastMonitoringThread::MCOUNT;
    fmt_.m_data.ministateBins_=0;
    fmt_.m_data.microstateBins_ = 0; 
 
    //TODO: we could do fastpath output even before seeing lumi
    lastGlobalLumi_=0;//this means no fast path before begingGlobalLumi (for now), 
    isGlobalLumiTransition_=true;
    lumiFromSource_=0;

    //startup monitoring
    fmt_.resetFastMonitor(microstateDefPath_);
    fmt_.jsonMonitor_->setNStreams(nStreams_);
    fmt_.m_data.registerVariables(fmt_.jsonMonitor_.get(), nStreams_, threadIDAvailable_ ? nThreads_:0);
    std::atomic_thread_fence(std::memory_order_acquire);
    fmt_.start(&FastMonitoringService::dowork,this);
  }

  void FastMonitoringService::jobFailure()
  {
    macrostate_ = FastMonitoringThread::sError;
  }



  //new output module name is stream
  void FastMonitoringService::preModuleBeginJob(const edm::ModuleDescription& desc)
  {
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
    //std::cout << "path legenda*****************" << std::endl;
   // std::cout << makePathLegenda()   << std::endl;
    std::cout << "module legenda***************" << std::endl;
    std::cout << makeModuleLegenda() << std::endl;
    macrostate_ = FastMonitoringThread::sJobReady;

    //update number of entries in module histogram
    fmt_.m_data.microstateBins_ = encModule_.vecsize(); 
  }

  void FastMonitoringService::postEndJob()
  {
    macrostate_ = FastMonitoringThread::sJobEnded;
    fmt_.stop();
  }


  //this is gone
  void FastMonitoringService::prePathBeginRun(const std::string& pathName)
  {
    return ;
    //bonus track, now monitoring path execution too...
    // here we are forced to use string keys...
//    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>update path map with " << pathName << std::endl;
//    encPath_.update((void*)&pathName);
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


	  lumiStartTime_[newLumi]=lumiStartTime;
	  while (!lastGlobalLumisClosed_.empty()) {
		  //wipe out old map entries as they aren't needed and slow down access
		  unsigned int oldLumi = lastGlobalLumisClosed_.back();
		  lastGlobalLumisClosed_.pop();
		  lumiStartTime_.erase(oldLumi);
		  //throughput_.erase(oldLumi);
		  avgLeadTime_.erase(oldLumi);
		  filesProcessedDuringLumi_.erase(oldLumi);
		  accuSize_.erase(oldLumi);
		  processedEventsPerLumi_.erase(oldLumi);
		  //streamEoLMap_.erase(oldLumi);
	  }
	  lastGlobalLumi_= newLumi;
	  isGlobalLumiTransition_=false;

	  //put a map for streams to report if they had EOL (else we have to update some variables)
	  //std::vector<bool> streamEolVec_;
	  //for (unsigned int i=0;i<nStreams_;i++) streamEolVec_.push_back(false);
	  //streamEoLMap_[newLumi] = streamEolVec_;

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
	  timeval stt = lumiStartTime_[lumi];
	  unsigned long usecondsForLumi = (lumiStopTime.tv_sec - stt.tv_sec)*1000000
		                  + (lumiStopTime.tv_usec - stt.tv_usec);
	  unsigned long accuSize = accuSize_.find(lumi)==accuSize_.end() ? 0 : accuSize_[lumi];
	  double throughput = throughputFactor()* double(accuSize) / double(usecondsForLumi);
	  //store to registered variable
	  fmt_.m_data.fastThroughputJ_.value() = throughput;

	  //update
	  doSnapshot(false,lumi,true,false,0);

	  // create file name for slow monitoring file
	  std::stringstream slowFileName;
	  slowFileName << slowName_ << "_ls" << std::setfill('0') << std::setw(4)
			<< lumi << "_pid" << std::setfill('0')
			<< std::setw(5) << getpid() << ".jsn";
	  boost::filesystem::path slow = workingDirectory_;
	  slow /= slowFileName.str();

	  //retrieve one result we need (todo: sanity check if it's found)
	  IntJ *lumiProcessedJptr = dynamic_cast<IntJ*>(fmt_.jsonMonitor_->getMergedIntJForLumi("Processed",lumi));
          assert(lumiProcessedJptr!=nullptr);
	  processedEventsPerLumi_[lumi] = lumiProcessedJptr->value();

	  //cross check (debugging)
	  {
	    auto itr = sourceEventsReport_.find(lumi);
	    if (itr==sourceEventsReport_.end()) {
              std::cout << "ERROR: SOURCE REPORT did not update yet for lumi" << lumi << std::endl;
	    }
	    else {
	      if (itr->second!=processedEventsPerLumi_[lumi]) {
		std::cout << " ERROR: MISMATCH with SOURCE REPORT from lumi" << lumi << "events(processed):" << processedEventsPerLumi_[lumi]
                          << " events(source):" << itr->second << std::endl;
		assert(0);//DEBUG!
	      }
	      sourceEventsReport_.erase(itr);
	    }
	  }

	  std::cout
			<< ">>> >>> FastMonitoringService: processed event count for this lumi = "
			<< lumiProcessedJptr->value() << " time = " << usecondsForLumi/1000000
			<< " size = " << accuSize << " thr = " << throughput << std::endl;
	  delete lumiProcessedJptr;

	  //full global and stream merge&output for this lumi
	  fmt_.jsonMonitor_->outputFullJSON(slow.string(),lumi);//full global and stream merge and JSON write for this lumi
	  fmt_.jsonMonitor_->discardCollected(lumi);//we don't do further updates for this lumi

	  isGlobalLumiTransition_=true;
	  fmt_.monlock_.unlock();
  }

  void FastMonitoringService::postGlobalEndLumi(edm::GlobalContext const& gc)
  {
    std::cout << "FastMonitoringService: Post-global-end LUMI: " << gc.luminosityBlockID().luminosityBlock() << std::endl;
    //mark closed lumis (still keep map entries until next one)
    lastGlobalLumisClosed_.push(gc.luminosityBlockID().luminosityBlock());
  }

  void FastMonitoringService::preStreamBeginLumi(edm::StreamContext const& sc)
  {
    unsigned int sid = sc.streamID().value();
    fmt_.monlock_.lock();
    fmt_.m_data.streamLumi_[sid] = sc.eventID().luminosityBlock();

    //reset collected values for this stream

    //instead of setting to 0, do atomic subtract (for consistent counting in case of data races)
    *(fmt_.m_data.processed_[sid])=0;
    //assuming the worst: that beginLumi runs in different thread than endLumi and does not yet see postEvent changes

    //unsigned int val = fmt_.m_data.processed_[sid]->load(std::memory_order_consume);
//    while (streamCounterUpdating_[sid]->load(std::memory_order_acquire)) {}
//    unsigned int val = fmt_.m_data.processed_[sid]->load(std::memory_order_relaxed);

//    fmt_.m_data.processed_[sid]->fetch_sub(val,std::memory_order_release);
    ministate_[sid]=&nopath_;
    microstate_[sid]=&reservedMicroStateNames[mInvalid];
    //#if TBB_IMPLEMENT_CPP0X
    //threadMicrostate_[tbb::thread::id()]=&reservedMicroStateNames[mInvalid];
    //#endif
    fmt_.monlock_.unlock();
  }

  void FastMonitoringService::preStreamEndLumi(edm::StreamContext const& sc)
  {
    unsigned int sid = sc.streamID().value();
    fmt_.monlock_.lock();
#if ATOMIC_LEVEL>=2
    //spinlock to make sure we are not still updating event counter somewhere
    while (streamCounterUpdating_[sid]->load(std::memory_order_acquire)) {}
#endif
    //update processed count to be complete at this time
    doStreamEOLSnapshot(false,sc.eventID().luminosityBlock(),sid);
    //reset this in case stream does not get notified of next lumi (we keep processed events only)
    ministate_[sid]=&nopath_;
    microstate_[sid]=&reservedMicroStateNames[mInvalid];
    //#if TBB_IMPLEMENT_CPP0X
    //threadMicrostate_[tbb::thread::id()]=&reservedMicroStateNames[mInvalid];
    //#endif
    fmt_.monlock_.unlock();
  }


  void FastMonitoringService::prePathEvent(edm::StreamContext const& sc, edm::PathContext const& pc)
  {
    //make sure that all path names are retrieved before allowing ministate to change
    //hack: assume memory is synchronized after ~50 events seen by each stream
    if (unlikely(eventCountForPathInit_[sc.streamID()]<50) && false==collectedPathList_[sc.streamID()]->load(std::memory_order_acquire))
    {
      initPathsLock_.lock();
      if (firstEventId_[sc.streamID()]==0) 
	firstEventId_[sc.streamID()]=sc.eventID().event();
      if (sc.eventID().event()==firstEventId_[sc.streamID()])
      {
	encPath_[sc.streamID()].update((void*)&pc.pathName());
	initPathsLock_.unlock();
	return;
      }
      else {
	collectedPathList_[sc.streamID()]->store(true,std::memory_order_seq_cst);
        fmt_.m_data.ministateBins_=encPath_[sc.streamID()].vecsize();
	initPathsLock_.unlock();
	//print paths
	//finished collecting path names
	std::cout << "path legenda*****************" << std::endl;
	std::cout << makePathLegenda()   << std::endl;
      }
    }
    else {
      ministate_[sc.streamID()] = &(pc.pathName());
    }
  }


  void FastMonitoringService::preEvent(edm::StreamContext const& sc)
  {
  }

  //shoudl be synchronized before
  void FastMonitoringService::postEvent(edm::StreamContext const& sc)
  {
    microstate_[sc.streamID()] = &reservedMicroStateNames[mFwkOvh];

    //#if TBB_IMPLEMENT_CPP0X
    //threadMicrostate_[tbb::thread::id()] = &reservedMicroStateNames[mFwkOvh];
    //#endif


    ministate_[sc.streamID()] = &nopath_;
    //fmt_.monlock_.lock();
#if ATOMIC_LEVEL>=2
    //use atomic flag to make sure end of lumi sees this
    streamCounterUpdating_[sc.streamID()]->store(true,std::memory_order_release);
    fmt_.m_data.processed_[sc.streamID()]->fetch_add(1,std::memory_order_release);
    streamCounterUpdating_[sc.streamID()]->store(false,std::memory_order_release);
#elif ATOMIC_LEVEL==1
    //writes are atomic, we assume writes propagate to memory before stream EOL snap
    fmt_.m_data.processed_[sc.streamID()]->fetch_add(1,std::memory_order_relaxed);
#elif ATOMIC_LEVEL==0
    (*(fmt_.m_data.processed_[sc.streamID()]))++;
#endif
    eventCountForPathInit_[sc.streamID()]++;
    //fmt_.monlock_.unlock();
  }

  void FastMonitoringService::preSourceEvent(edm::StreamID sid)
  {
    microstate_[sid.value()] = &reservedMicroStateNames[mIdle];

    //#if TBB_IMPLEMENT_CPP0X
    //threadMicrostate_[tbb::thread::id()] = &reservedMicroStateNames[mIdle];
    //#endif

  }

  void FastMonitoringService::postSourceEvent(edm::StreamID sid)
  {
    microstate_[sid.value()] = &reservedMicroStateNames[mFwkOvh];

    //#if TBB_IMPLEMENT_CPP0X
    //threadMicrostate_[tbb::thread::id()] = &reservedMicroStateNames[mFwkOvh];
    //#endif

  }

  void FastMonitoringService::preModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc)
  {
    microstate_[sc.streamID().value()] = (void*)(mcc.moduleDescription());
    //#if TBB_IMPLEMENT_CPP0X
    //threadMicrostate_[tbb::thread::id()] = (void*)(mcc.moduleDescription());
    //#endif
  }

  void FastMonitoringService::postModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc)
  {
    microstate_[sc.streamID().value()] = (void*)(mcc.moduleDescription());

    //#if TBB_IMPLEMENT_CPP0X
    //threadMicrostate_[tbb::thread::id()] = (void*)(mcc.moduleDescription());
    //#endif

    //maybe this should be:
    //microstate_[sc.streamID().value()] = &reservedMicroStateNames[mFwkOvh];
  }

  //FUNCTIONS CALLED FROM OUTSIDE

  //this is for old-fashioned service that is not thread safe and can block other streams
  //(we assume the worst case - everything is blocked)
  void FastMonitoringService::setMicroState(MicroStateService::Microstate m)
  {
    for (unsigned int i=0;i<nStreams_;i++)
      microstate_[i] = &reservedMicroStateNames[m];
    //#if TBB_IMPLEMENT_CPP0X
    //for (unsigned int i=0;i<nThreads_;i++)
    //  threadMicrostate_[i] = &reservedMicroStateNames[m];
    //#endif
  }

  //this is for another service that is thread safe or rarely blocks other streams
  void FastMonitoringService::setMicroState(edm::StreamID sid, MicroStateService::Microstate m)
  {
    microstate_[sid] = &reservedMicroStateNames[m];
    //#if TBB_IMPLEMENT_CPP0X
    //threadMicrostate_[tbb::thread::id()] = &reservedMicroStateNames[m];
    //#endif
  }

  //from source - needs changes to source to track per-lumi file processing
  void FastMonitoringService::accumulateFileSize(unsigned int lumi, unsigned long fileSize) {
	fmt_.monlock_.lock();
	if (accuSize_.find(lumi)==accuSize_.end()) accuSize_[lumi] = fileSize;
	else accuSize_[lumi] += fileSize;

	if (filesProcessedDuringLumi_.find(lumi)==filesProcessedDuringLumi_.end())
	  filesProcessedDuringLumi_[lumi] = 1;
	else
	  filesProcessedDuringLumi_[lumi]++;
	fmt_.monlock_.unlock();
  }

  //should be measured by the source and provided when found
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
	  fmt_.monlock_.lock();
	  if (lumi>lumiFromSource_) {
		  lumiFromSource_=lumi;
		  leadTimes_.clear();
	  }
	  //TODO:improve precision?
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
	  fmt_.monlock_.unlock();
  }

  //for the output module
  unsigned int FastMonitoringService::getEventsProcessedForLumi(unsigned int lumi) {
	  fmt_.monlock_.lock();
	  auto it = processedEventsPerLumi_.find(lumi);
	  if (it!=processedEventsPerLumi_.end()) {
		  unsigned int proc = it->second;
		  fmt_.monlock_.unlock();
		  return proc;
	  }
	  else {
		  fmt_.monlock_.unlock();
		  std::cout << "ERROR: output module wants deleted info " << std::endl;
		  assert(0);//DEBUG!
		  return 0;
	  }
  }


  void FastMonitoringService::doSnapshot(bool outputCSV, unsigned int forLumi, bool isGlobalEOL, bool isStream, unsigned int streamID)
  {

    // update macrostate
    fmt_.m_data.fastMacrostateJ_ = macrostate_;

    //update these unless in the midst of a global transition
    if (!isGlobalLumiTransition_) {

      auto itd = avgLeadTime_.find(forLumi);
      if (itd != avgLeadTime_.end()) 
	fmt_.m_data.fastAvgLeadTimeJ_ = itd->second;
      else fmt_.m_data.fastAvgLeadTimeJ_=0.;

      auto iti = filesProcessedDuringLumi_.find(forLumi);
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
    //#if TBB_IMPLEMENT_CPP0X
    //for (unsigned int i=0;i<nThreads_;i++)
    //  fmt_.m_data.threadMicrostateEncoded_[i] = encModule_.encode(threadMicrostate_[i]);
    //#endif
    
    if (isGlobalEOL)
    {//only update global variables
      fmt_.jsonMonitor_->snapGlobal(outputCSV, fastPath_,forLumi);
    }
    else
      fmt_.jsonMonitor_->snap(outputCSV, fastPath_,forLumi);
  }

  void FastMonitoringService::reportEventsThisLumiInSource(unsigned int lumi,unsigned int events)
  {

      fmt_.monlock_.lock();
      auto itr = sourceEventsReport_.find(lumi);
      if (itr!=sourceEventsReport_.end())
        sourceEventsReport_[lumi]+=events;
      else 
      sourceEventsReport_[lumi]=events;

      fmt_.monlock_.unlock();
  }
} //end namespace evf

