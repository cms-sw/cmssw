////////////////////////////////////////////////////////////////////////////////
//
// FUEventProcessor
// ----------------
//
////////////////////////////////////////////////////////////////////////////////

#include "FUEventProcessor.h"
#include "procUtils.h"
#include "EventFilter/Utilities/interface/CPUStat.h"
#include "EventFilter/Utilities/interface/RateStat.h"

#include "EventFilter/Utilities/interface/Exception.h"

#include "EventFilter/Message2log4cplus/interface/MLlog4cplus.h"
#include "EventFilter/Modules/interface/FUShmDQMOutputService.h"
#include "EventFilter/Utilities/interface/ServiceWebRegistry.h"
#include "EventFilter/Utilities/interface/ServiceWeb.h"
#include "EventFilter/Utilities/interface/ModuleWeb.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "EventFilter/Modules/src/FUShmOutputModule.h"

#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "toolbox/BSem.h" 
#include "toolbox/Runtime.h"
#include "toolbox/stacktrace.h"
#include "toolbox/net/Utils.h"

#include <boost/tokenizer.hpp>

#include "xcept/tools.h"
#include "xgi/Method.h"

#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"


#include <sys/wait.h>
#include <sys/utsname.h>
#include <sys/mman.h>
#include <signal.h>

#include <typeinfo>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>


using namespace evf;
using namespace cgicc;
namespace toolbox {
  namespace mem {
    extern toolbox::BSem * _s_mutex_ptr_; 
  }
}

//signal handler (global)
namespace evf {
  FUEventProcessor * FUInstancePtr_;
  int evfep_raised_signal;
  void evfep_sighandler(int sig, siginfo_t* info, void* c)
  {
    evfep_raised_signal=sig;
    FUInstancePtr_->handleSignalSlave(sig, info, c);
  }
  void evfep_alarmhandler(int sig, siginfo_t* info, void* c)
  {
    if (evfep_raised_signal) {
      signal(evfep_raised_signal,SIG_DFL);
      raise(evfep_raised_signal);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUEventProcessor::FUEventProcessor(xdaq::ApplicationStub *s) 
  : xdaq::Application(s)
  , fsm_(this)
  , log_(getApplicationLogger())
  , evtProcessor_(log_, getApplicationDescriptor()->getInstance())
  , runNumber_(0)
  , epInitialized_(false)
  , outPut_(true)
  , autoRestartSlaves_(false)
  , slaveRestartDelaySecs_(10)
  , hasShMem_(true)
  , hasPrescaleService_(true)
  , hasModuleWebRegistry_(true)
  , hasServiceWebRegistry_(true)
  , isRunNumberSetter_(true)
  , iDieStatisticsGathering_(false)
  , outprev_(true)
  , exitOnError_(true)
  , reasonForFailedState_()
  , squidnet_(3128,"http://localhost:8000/RELEASE-NOTES.txt")
  , logRing_(logRingSize_)
  , logRingIndex_(logRingSize_)
  , logWrap_(false)
  , nbSubProcesses_(0)
  , nbSubProcessesReporting_(0)
  , forkInEDM_(true)
  , nblive_(0)
  , nbdead_(0)
  , nbTotalDQM_(0)
  , wlReceiving_(0)
  , asReceiveMsgAndExecute_(0)
  , receiving_(false) 
  , wlReceivingMonitor_(0)
  , asReceiveMsgAndRead_(0)
  , receivingM_(false)
  , myProcess_(0)
  , wlSupervising_(0)
  , asSupervisor_(0)
  , supervising_(false)
  , wlSignalMonitor_(0)
  , asSignalMonitor_(0)
  , signalMonitorActive_(false)
  , monitorInfoSpace_(0)
  , monitorLegendaInfoSpace_(0)
  , applicationInfoSpace_(0)
  , nbProcessed(0)
  , nbAccepted(0)
  , scalersInfoSpace_(0)
  , scalersLegendaInfoSpace_(0)
  , wlScalers_(0)
  , asScalers_(0)
  , wlScalersActive_(false)
  , scalersUpdates_(0)
  , wlSummarize_(0)
  , asSummarize_(0)
  , wlSummarizeActive_(false)
  , superSleepSec_(1)
  , iDieUrl_("none")
  , vulture_(0)
  , vp_(0)
  , cpustat_(0)
  , ratestat_(0)
  , mwrRef_(nullptr)
  , sorRef_(nullptr)
  , master_message_prg_(0,MSQM_MESSAGE_TYPE_PRG)
  , master_message_prr_(MAX_MSG_SIZE,MSQS_MESSAGE_TYPE_PRR)
  , slave_message_prr_(sizeof(prg),MSQS_MESSAGE_TYPE_PRR)
  , master_message_trr_(MAX_MSG_SIZE,MSQS_MESSAGE_TYPE_TRR)
  , edm_init_done_(true)
  , crashesThisRun_(false)
  , rlimit_coresize_changed_(false)
  , crashesToDump_(2)
  , sigmon_sem_(0)
  , datasetCounting_(true)
{
  using namespace utils;

  FUInstancePtr_=this;

  names_.push_back("nbProcessed"    );
  names_.push_back("nbAccepted"     );
  names_.push_back("epMacroStateInt");
  names_.push_back("epMicroStateInt");
  // create pipe for web communication
  int retpipe = pipe(anonymousPipe_);
  if(retpipe != 0)
        LOG4CPLUS_ERROR(getApplicationLogger(),"Failed to create pipe");
  // check squid presence
  squidPresent_ = squidnet_.check();
  //pass application parameters to FWEPWrapper
  evtProcessor_.setAppDesc(getApplicationDescriptor());
  evtProcessor_.setAppCtxt(getApplicationContext());
  // bind relevant callbacks to finite state machine
  fsm_.initialize<evf::FUEventProcessor>(this);
  
  //set sourceId_
  url_ =
    getApplicationDescriptor()->getContextDescriptor()->getURL()+"/"+
    getApplicationDescriptor()->getURN();
  class_   =getApplicationDescriptor()->getClassName();
  instance_=getApplicationDescriptor()->getInstance();
  sourceId_=class_.toString()+"_"+instance_.toString();
  LOG4CPLUS_INFO(getApplicationLogger(),sourceId_ <<" constructor"         );
  LOG4CPLUS_INFO(getApplicationLogger(),"CMSSW_BASE:"<<getenv("CMSSW_BASE"));
  
  getApplicationDescriptor()->setAttribute("icon", "/evf/images/epicon.jpg");
  
  xdata::InfoSpace *ispace = getApplicationInfoSpace();
  applicationInfoSpace_ = ispace;

  // default configuration
  ispace->fireItemAvailable("parameterSet",         &configString_                );
  ispace->fireItemAvailable("epInitialized",        &epInitialized_               );
  ispace->fireItemAvailable("stateName",             fsm_.stateName()             );
  ispace->fireItemAvailable("runNumber",            &runNumber_                   );
  ispace->fireItemAvailable("outputEnabled",        &outPut_                      );

  ispace->fireItemAvailable("hasSharedMemory",      &hasShMem_);
  ispace->fireItemAvailable("hasPrescaleService",   &hasPrescaleService_          );
  ispace->fireItemAvailable("hasModuleWebRegistry", &hasModuleWebRegistry_        );
  ispace->fireItemAvailable("hasServiceWebRegistry", &hasServiceWebRegistry_      );
  ispace->fireItemAvailable("isRunNumberSetter",    &isRunNumberSetter_           );
  ispace->fireItemAvailable("iDieStatisticsGathering",   &iDieStatisticsGathering_);
  ispace->fireItemAvailable("rcmsStateListener",     fsm_.rcmsStateListener()     );
  ispace->fireItemAvailable("foundRcmsStateListener",fsm_.foundRcmsStateListener());
  ispace->fireItemAvailable("nbSubProcesses",       &nbSubProcesses_              );
  ispace->fireItemAvailable("nbSubProcessesReporting",&nbSubProcessesReporting_   );
  ispace->fireItemAvailable("forkInEDM"             ,&forkInEDM_                  );
  ispace->fireItemAvailable("superSleepSec",        &superSleepSec_               );
  ispace->fireItemAvailable("autoRestartSlaves",    &autoRestartSlaves_           );
  ispace->fireItemAvailable("slaveRestartDelaySecs",&slaveRestartDelaySecs_       );
  ispace->fireItemAvailable("iDieUrl",              &iDieUrl_                     );
  ispace->fireItemAvailable("crashesToDump"         ,&crashesToDump_              );
  ispace->fireItemAvailable("datasetCounting"       ,&datasetCounting_            );

  // Add infospace listeners for exporting data values
  getApplicationInfoSpace()->addItemChangedListener("parameterSet",        this);
  getApplicationInfoSpace()->addItemChangedListener("outputEnabled",       this);

  // findRcmsStateListener
  fsm_.findRcmsStateListener();
  
  // initialize monitoring infospace

  std::string monInfoSpaceName="evf-eventprocessor-status-monitor";
  toolbox::net::URN urn = this->createQualifiedInfoSpace(monInfoSpaceName);
  monitorInfoSpace_ = xdata::getInfoSpaceFactory()->get(urn.toString());

  std::string monLegendaInfoSpaceName="evf-eventprocessor-status-legenda";
  urn = this->createQualifiedInfoSpace(monLegendaInfoSpaceName);
  monitorLegendaInfoSpace_ = xdata::getInfoSpaceFactory()->get(urn.toString());

  
  monitorInfoSpace_->fireItemAvailable("url",                      &url_            );
  monitorInfoSpace_->fireItemAvailable("class",                    &class_          );
  monitorInfoSpace_->fireItemAvailable("instance",                 &instance_       );
  monitorInfoSpace_->fireItemAvailable("runNumber",                &runNumber_      );
  monitorInfoSpace_->fireItemAvailable("stateName",                 fsm_.stateName()); 

  monitorInfoSpace_->fireItemAvailable("squidPresent",             &squidPresent_   );

  std::string scalersInfoSpaceName="evf-eventprocessor-scalers-monitor";
  urn = this->createQualifiedInfoSpace(scalersInfoSpaceName);
  scalersInfoSpace_ = xdata::getInfoSpaceFactory()->get(urn.toString());

  std::string scalersLegendaInfoSpaceName="evf-eventprocessor-scalers-legenda";
  urn = this->createQualifiedInfoSpace(scalersLegendaInfoSpaceName);
  scalersLegendaInfoSpace_ = xdata::getInfoSpaceFactory()->get(urn.toString());



  evtProcessor_.setScalersInfoSpace(scalersInfoSpace_,scalersLegendaInfoSpace_);
  scalersInfoSpace_->fireItemAvailable("instance", &instance_);

  evtProcessor_.setApplicationInfoSpace(ispace);
  evtProcessor_.setMonitorInfoSpace(monitorInfoSpace_,monitorLegendaInfoSpace_);
  evtProcessor_.publishConfigAndMonitorItems(nbSubProcesses_.value_!=0);

  //subprocess state vectors for MP
  monitorInfoSpace_->fireItemAvailable("epMacroStateInt",             &spMStates_); 
  monitorInfoSpace_->fireItemAvailable("epMicroStateInt",             &spmStates_); 
  
  // Bind web interface
  xgi::bind(this, &FUEventProcessor::css,              "styles.css");
  xgi::bind(this, &FUEventProcessor::defaultWebPage,   "Default"   );
  xgi::bind(this, &FUEventProcessor::spotlightWebPage, "Spotlight" );
  xgi::bind(this, &FUEventProcessor::scalersWeb,       "scalersWeb");
  xgi::bind(this, &FUEventProcessor::pathNames,        "pathNames" );
  xgi::bind(this, &FUEventProcessor::subWeb,           "SubWeb"    );
  xgi::bind(this, &FUEventProcessor::getSlavePids,     "getSlavePids");
  xgi::bind(this, &FUEventProcessor::moduleWeb,        "moduleWeb" );
  xgi::bind(this, &FUEventProcessor::serviceWeb,       "serviceWeb");
  xgi::bind(this, &FUEventProcessor::microState,       "microState");
  xgi::bind(this, &FUEventProcessor::updater,          "updater"   );
  xgi::bind(this, &FUEventProcessor::procStat,         "procStat"  );

  // instantiate the plugin manager, not referenced here after!

  edm::AssertHandler ah;

  //create Message Service thread and pass ownership to auto_ptr that we destroy before fork
  try{
    LOG4CPLUS_DEBUG(getApplicationLogger(),
		    "Trying to create message service presence ");
    edm::PresenceFactory *pf = edm::PresenceFactory::get();
    if(pf != 0) {
      messageServicePresence_= pf->makePresence("MessageServicePresence");
    }
    else {
      LOG4CPLUS_ERROR(getApplicationLogger(),
		      "Unable to create message service presence ");
    }
  } 
  catch(...) {
    LOG4CPLUS_ERROR(getApplicationLogger(),"Unknown Exception");
  }
  ML::MLlog4cplus::setAppl(this);      

  typedef std::set<xdaq::ApplicationDescriptor*> AppDescSet_t;
  typedef AppDescSet_t::iterator                 AppDescIter_t;
  
  AppDescSet_t rcms=
    getApplicationContext()->getDefaultZone()->
    getApplicationDescriptors("RCMSStateListener");
  if(rcms.size()==0) 
    {
      LOG4CPLUS_WARN(getApplicationLogger(),
		       "MonitorReceiver not found, perhaphs it has not been defined ? Scalers updater wl will bail out!");
      //	localLog("-W- MonitorReceiver not found, perhaphs it has not been defined ? Scalers updater wl will bail out!");
    }
  else
    {
      AppDescIter_t it = rcms.begin();
      evtProcessor_.setRcms(*it);
    }
  pthread_mutex_init(&start_lock_,0);
  pthread_mutex_init(&stop_lock_,0);
  pthread_mutex_init(&pickup_lock_,0);

  forkInfoObj_=nullptr;
  pthread_mutex_init(&forkObjLock_,0);
  makeStaticInfo();
  startSupervisorLoop();  

  if(vulture_==0) vulture_ = new Vulture(true);

  ////////////////////////////////  

  AppDescSet_t setOfiDie=
    getApplicationContext()->getDefaultZone()->
    getApplicationDescriptors("evf::iDie");
  
  for (AppDescIter_t it=setOfiDie.begin();it!=setOfiDie.end();++it)
    if ((*it)->getInstance()==0) // there has to be only one instance of iDie
      iDieUrl_ = (*it)->getContextDescriptor()->getURL() + "/" + (*it)->getURN();

  //save default core file size
  getrlimit(RLIMIT_CORE,&rlimit_coresize_default_);

  //prepare IPC semaphore for getting the workloop waked up on signal caught in slaves
  #ifdef linux
  if (sigmon_sem_==0) {
    sigmon_sem_ = (sem_t*)mmap(NULL, sizeof(sem_t),
	PROT_READ | PROT_WRITE,
	MAP_ANONYMOUS | MAP_SHARED, 0, 0);
    if (!sigmon_sem_) {
      perror("mmap error\n");
      std::cout << "mmap error"<<std::endl;
    }
    else
      sem_init(sigmon_sem_,true,0);
  }
  #endif
}
//___________here ends the *huge* constructor___________________________________


//______________________________________________________________________________
FUEventProcessor::~FUEventProcessor()
{
  // no longer needed since the wrapper is a member of the class and one can rely on 
  // implicit destructor - to be revised - at any rate the most common exit path is via "kill"...
  //  if (evtProcessor_) delete evtProcessor_;
  if(vulture_ != 0) delete vulture_;
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////


//______________________________________________________________________________
bool FUEventProcessor::configuring(toolbox::task::WorkLoop* wl)
{
//   std::cout << "values " << ((nbSubProcesses_.value_!=0) ? 0x10 : 0) << " "
// 	    << ((instance_.value_==0) ? 0x8 : 0) << " "
// 	    << (hasServiceWebRegistry_.value_ ? 0x4 : 0) << " "
// 	    << (hasModuleWebRegistry_.value_ ? 0x2 : 0) << " "
// 	    << (hasPrescaleService_.value_ ? 0x1 : 0) <<std::endl;
  unsigned short smap 
    = (datasetCounting_.value_ ? 0x20 : 0 )
    + ((nbSubProcesses_.value_!=0) ? 0x10 : 0)
    + (((instance_.value_%80)==0) ? 0x8 : 0) // have at least one legend per slice
    + (hasServiceWebRegistry_.value_ ? 0x4 : 0) 
    + (hasModuleWebRegistry_.value_ ? 0x2 : 0) 
    + (hasPrescaleService_.value_ ? 0x1 : 0);
  if(nbSubProcesses_.value_==0) 
    {
      spMStates_.setSize(1); 
      spmStates_.setSize(1); 
    }
  else
    {
      spMStates_.setSize(nbSubProcesses_.value_);
      spmStates_.setSize(nbSubProcesses_.value_);
      for(unsigned int i = 0; i < spMStates_.size(); i++)
	{
	  spMStates_[i] = edm::event_processor::sInit; 
	  spmStates_[i] = 0; 
	}
    }
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start configuring ...");
    std::string cfg = configString_.toString(); evtProcessor_.init(smap,cfg);
    epInitialized_=true;
    if(evtProcessor_)
      {
	//get ref of mwr
        mwrRef_ = evtProcessor_.getModuleWebRegistry();
        sorRef_ = evtProcessor_.getShmOutputModuleRegistry();
	// moved to wrapper class
	configuration_ = evtProcessor_.configuration();
	if(nbSubProcesses_.value_==0) evtProcessor_.startMonitoringWorkLoop(); 
	evtProcessor_->beginJob();
	evtProcessor_.setupFastTimerService(nbSubProcesses_.value_>0 ? nbSubProcesses_.value_:1);
	if(cpustat_) {delete cpustat_; cpustat_=0;}
	cpustat_ = new CPUStat(evtProcessor_.getNumberOfMicrostates(),
			       nbSubProcesses_.value_,
			       instance_.value_,
			       iDieUrl_.value_);
	if(ratestat_) {delete ratestat_; ratestat_=0;}
	ratestat_ = new RateStat(iDieUrl_.value_);
	if(iDieStatisticsGathering_.value_)
	{
	  try
	  {
	    cpustat_->sendLegenda(evtProcessor_.getmicromap());
	    xdata::Serializable *legenda = scalersLegendaInfoSpace_->find("scalersLegenda");
	    if(legenda !=0)
	    {
	      std::string slegenda = ((xdata::String*)legenda)->value_;
	      ratestat_->sendLegenda(slegenda);
	    }
	    if (sorRef_ && datasetCounting_.value_)
	    {
	      xdata::String dsLegenda =  sorRef_->getDatasetCSV();
	      if (dsLegenda.value_.size())
	        ratestat_->sendAuxLegenda(dsLegenda);
	    }
	  }
	  catch(evf::Exception &e)
	  {
	    LOG4CPLUS_INFO(getApplicationLogger(),"coud not send legenda"
		<< e.what());
	  }
	  catch (xcept::Exception& e) {
	    LOG4CPLUS_ERROR(getApplicationLogger(),"Failed to get or send legenda."
		<< e.what());
	  }
	}

	fsm_.fireEvent("ConfigureDone",this);
	LOG4CPLUS_INFO(getApplicationLogger(),"Finished configuring!");
	localLog("-I- Configuration completed");

      }
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "configuring FAILED: " + (std::string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    localLog(reasonForFailedState_);
  }
  catch(cms::Exception &e) {
    reasonForFailedState_ = e.explainSelf();
    fsm_.fireFailed(reasonForFailedState_,this);
    localLog(reasonForFailedState_);
  }    
  catch(std::exception &e) {
    reasonForFailedState_ = e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    localLog(reasonForFailedState_);
  }
  catch(...) {
    fsm_.fireFailed("Unknown Exception",this);
  }

  if(vulture_!=0 && vp_ == 0) vp_ = vulture_->makeProcess();

  return false;
}




//______________________________________________________________________________
bool FUEventProcessor::enabling(toolbox::task::WorkLoop* wl)
{
  nbTotalDQM_ = 0;
  scalersUpdates_ = 0;
  idleProcStats_ = 0;
  allProcStats_ = 0;
//   std::cout << "values " << ((nbSubProcesses_.value_!=0) ? 0x10 : 0) << " "
// 	    << ((instance_.value_==0) ? 0x8 : 0) << " "
// 	    << (hasServiceWebRegistry_.value_ ? 0x4 : 0) << " "
// 	    << (hasModuleWebRegistry_.value_ ? 0x2 : 0) << " "
// 	    << (hasPrescaleService_.value_ ? 0x1 : 0) <<std::endl;
  unsigned short smap 
    = (datasetCounting_.value_ ? 0x20 : 0 )
    + ((nbSubProcesses_.value_!=0) ? 0x10 : 0)
    + (((instance_.value_%80)==0) ? 0x8 : 0) // have at least one legend per slice
    + (hasServiceWebRegistry_.value_ ? 0x4 : 0) 
    + (hasModuleWebRegistry_.value_ ? 0x2 : 0) 
    + (hasPrescaleService_.value_ ? 0x1 : 0);

  LOG4CPLUS_INFO(getApplicationLogger(),"Start enabling...");
  
  //reset core limit size
  if (rlimit_coresize_changed_)
    setrlimit(RLIMIT_CORE,&rlimit_coresize_default_);
  rlimit_coresize_changed_=false;
  crashesThisRun_=0;
  //recreate signal monitor sem
  sem_destroy(sigmon_sem_);
  sem_init(sigmon_sem_,true,0);

  if(!epInitialized_){
    evtProcessor_.forceInitEventProcessorMaybe();
  }
  std::string cfg = configString_.toString(); evtProcessor_.init(smap,cfg);
  
  mwrRef_ = evtProcessor_.getModuleWebRegistry();
  sorRef_ = evtProcessor_.getShmOutputModuleRegistry();

  if(!epInitialized_){
    evtProcessor_->beginJob(); 
    evtProcessor_.setupFastTimerService(nbSubProcesses_.value_>0 ? nbSubProcesses_.value_:1);
    if(cpustat_) {delete cpustat_; cpustat_=0;}
    cpustat_ = new CPUStat(evtProcessor_.getNumberOfMicrostates(),
			   nbSubProcesses_.value_,
			   instance_.value_,
			   iDieUrl_.value_);
    if(ratestat_) {delete ratestat_; ratestat_=0;}
    ratestat_ = new RateStat(iDieUrl_.value_);
    if(iDieStatisticsGathering_.value_)
    {
      try
      {
	cpustat_->sendLegenda(evtProcessor_.getmicromap());
	xdata::Serializable *legenda = scalersLegendaInfoSpace_->find("scalersLegenda");
	if(legenda !=0)
	{
	  std::string slegenda = ((xdata::String*)legenda)->value_;
	  ratestat_->sendLegenda(slegenda);
	}
	if (sorRef_ && datasetCounting_.value_)
	{
	  xdata::String dsLegenda =  sorRef_->getDatasetCSV();
	  if (dsLegenda.value_.size())
	    ratestat_->sendAuxLegenda(dsLegenda);
	}
      }
      catch(evf::Exception &e)
      {
	LOG4CPLUS_INFO(getApplicationLogger(),"could not send legenda"
	    << e.what());
      }
      catch (xcept::Exception& e) {
	LOG4CPLUS_ERROR(getApplicationLogger(),"Failed to get or send legenda."
	    << e.what());
      }
    }
    epInitialized_ = true;
  }
  configuration_ = evtProcessor_.configuration(); // get it again after init has been carried out...
  evtProcessor_.resetLumiSectionReferenceIndex();
  //classic appl will return here 
  if(nbSubProcesses_.value_==0) return enableClassic();
  //protect manipulation of subprocess array
  pthread_mutex_lock(&start_lock_);
  pthread_mutex_lock(&pickup_lock_);
  subs_.clear();
  subs_.resize(nbSubProcesses_.value_); // this should not be necessary
  pid_t retval = -1;

  for(unsigned int i=0; i<nbSubProcesses_.value_; i++)
    {
      subs_[i]=SubProcess(i,retval); //this will replace all the scattered variables
    }
  pthread_mutex_unlock(&pickup_lock_);

  pthread_mutex_unlock(&start_lock_);

  //set expected number of child EP's for the Init message(s) sent to the SM
  try {
    if (sorRef_) {
      unsigned int nbExpectedEPWriters = nbSubProcesses_.value_;
      if (nbExpectedEPWriters==0) nbExpectedEPWriters=1;//master instance processing
      std::vector<edm::FUShmOutputModule *> shmOutputs = sorRef_->getShmOutputModules();
      for (unsigned int i=0;i<shmOutputs.size();i++) {
	shmOutputs[i]->setNExpectedEPs(nbExpectedEPWriters);
      }
    }
  }
  catch (...)
  {
    LOG4CPLUS_ERROR(getApplicationLogger(),"Thrown Exception while setting nExpectedEPs in shmOutputs");
  }

  //use new method if configured
  edm_init_done_=true;
  if (forkInEDM_.value_) {
    edm_init_done_=false;
    enableForkInEDM();
  }
  else
    for(unsigned int i=0; i<nbSubProcesses_.value_; i++)
    {
      retval = subs_[i].forkNew();
      if(retval==0)
	{
	  myProcess_ = &subs_[i];
	  // dirty hack: delete/recreate global binary semaphore for later use in child
	  delete toolbox::mem::_s_mutex_ptr_;
	  toolbox::mem::_s_mutex_ptr_ = new toolbox::BSem(toolbox::BSem::FULL,true);
	  int retval = pthread_mutex_destroy(&stop_lock_);
	  if(retval != 0) perror("error");
	  retval = pthread_mutex_init(&stop_lock_,0);
	  if(retval != 0) perror("error");
 	  fsm_.disableRcmsStateNotification();
	  
	  return enableMPEPSlave();
	  // the loop is broken in the child 
	}
    }

  if (forkInEDM_.value_) {

    edm::event_processor::State st;
    while (!edm_init_done_) {
      usleep(10000);
      st = evtProcessor_->getState();
      if (st==edm::event_processor::sError || st==edm::event_processor::sInvalid) break;
    }
    st = evtProcessor_->getState();
    //error handling: EP must fork during sRunning
    if (st!=edm::event_processor::sRunning) {
      reasonForFailedState_ = std::string("Master edm::EventProcessor in state ") + evtProcessor_->stateName(st);
      localLog(reasonForFailedState_);
      fsm_.fireFailed(reasonForFailedState_,this);
      return false;
    }

    sleep(1);
    startSummarizeWorkLoop();
    startSignalMonitorWorkLoop();//only with new forking
    vp_ = vulture_->start(iDieUrl_.value_,runNumber_.value_);

    //enable after we are done with conditions loading and forking
    LOG4CPLUS_INFO(getApplicationLogger(),"Finished enabling!");
    fsm_.fireEvent("EnableDone",this);
    localLog("-I- Start completed");
    return false;
  }

  startSummarizeWorkLoop();
  vp_ = vulture_->start(iDieUrl_.value_,runNumber_.value_);
  LOG4CPLUS_INFO(getApplicationLogger(),"Finished enabling!");
  fsm_.fireEvent("EnableDone",this);
  localLog("-I- Start completed");
  return false;
}

bool FUEventProcessor::doEndRunInEDM() {
  //taking care that EP in master is in stoppable state
  if (forkInfoObj_) {

    int count = 30;
    bool waitedForEDM=false;
    while (!edm_init_done_ && count) {
      ::sleep(1);
      if (count%5==0)
        LOG4CPLUS_WARN(log_,"MASTER EP: Stopping while EP busy in beginRun. waiting " <<count<< "sec");
      count--;
      waitedForEDM=true;
    }
    //sleep a few more seconds it was early stop
    if (waitedForEDM) sleep(5);

    //if (count==0) fsm_.fireFailed("failed to stop Master EP",this);

    if (evtProcessor_->getState()==edm::event_processor::sJobReady)
      return true;//we are already done

    forkInfoObj_->lock();
    forkInfoObj_->stopCondition=true;
    sem_post(forkInfoObj_->control_sem_);
    forkInfoObj_->unlock();

    count = 30;

    edm::event_processor::State st;
    while(count--) {
      st = evtProcessor_->getState();
      if (st==edm::event_processor::sRunning) ::usleep(100000);
      else if (st==edm::event_processor::sStopping || st==edm::event_processor::sJobReady) {
        break;
      }
      else {
	std::ostringstream ost;
        ost << "Master edm::EventProcessor is in state "<< evtProcessor_->stateName(st) << " while stopping";
        LOG4CPLUS_ERROR(getApplicationLogger(),ost.str());
        fsm_.fireFailed(ost.str(),this);
        return false;
      }
      if (count%5==0 && st==edm::event_processor::sRunning && !forkInfoObj_->receivedStop_) {
	forkInfoObj_->lock();
	forkInfoObj_->stopCondition=true;
	sem_post(forkInfoObj_->control_sem_);
	forkInfoObj_->unlock();
        LOG4CPLUS_WARN(getApplicationLogger(),
	  "Master edm::EventProcessor still running after "<< (30-count-1) << " seconds. \"sem_post\" was executed again" );
      }
    }
    if (count<0) {
      std::ostringstream ost;
      if (!forkInfoObj_->receivedStop_)
        ost << "Timeout waiting for Master edm::EventProcessor to go stopping state "
	    << evtProcessor_->stateName(st) << ": input source did not receive stop signal!";
      else
        ost << "Timeout waiting for Master edm::EventProcessor to go stopping state "<<evtProcessor_->stateName(st);
      LOG4CPLUS_ERROR(getApplicationLogger(),ost.str());
      fsm_.fireFailed(ost.str(),this);
      return false;
    }
  }
  return true;
}

//______________________________________________________________________________
bool FUEventProcessor::stopping(toolbox::task::WorkLoop* wl)
{
  setrlimit(RLIMIT_CORE,&rlimit_coresize_default_);
  if(nbSubProcesses_.value_!=0) {
    stopSlavesAndAcknowledge();
    if (forkInEDM_.value_) {
            //only in new forking for now
            sem_post(sigmon_sem_);
	    if (!doEndRunInEDM())
	      return false;
    }
  }
  vulture_->stop();

  if (forkInEDM_.value_) {
    //shared memory was already disconnected in master
    bool tmpHasShMem_=hasShMem_;
    hasShMem_=false;
    stopClassic();
    hasShMem_=tmpHasShMem_;
    return false;
  }
  stopClassic();
  return false;
}


//______________________________________________________________________________
bool FUEventProcessor::halting(toolbox::task::WorkLoop* wl)
{
  LOG4CPLUS_INFO(getApplicationLogger(),"Start halting ...");
  setrlimit(RLIMIT_CORE,&rlimit_coresize_default_);
  if(nbSubProcesses_.value_!=0) { 
    stopSlavesAndAcknowledge();
    if (forkInEDM_.value_) {
            sem_post(sigmon_sem_);
	    if (!doEndRunInEDM())
	      return false;

    }
  }
  try{
    evtProcessor_.stopAndHalt();
    //cleanup forking variables
    if (forkInfoObj_) {
      delete forkInfoObj_;
      forkInfoObj_=0;
    }
  }
  catch (evf::Exception &e) {
    reasonForFailedState_ = "halting FAILED: " + (std::string)e.what();
    localLog(reasonForFailedState_);
    fsm_.fireFailed(reasonForFailedState_,this);
  }
  //  if(hasShMem_) detachDqmFromShm();

  LOG4CPLUS_INFO(getApplicationLogger(),"Finished halting!");
  fsm_.fireEvent("HaltDone",this);

  localLog("-I- Halt completed");
  return false;
}


//______________________________________________________________________________
xoap::MessageReference FUEventProcessor::fsmCallback(xoap::MessageReference msg)
  throw (xoap::exception::Exception)
{
  return fsm_.commandCallback(msg);
}


//______________________________________________________________________________
void FUEventProcessor::actionPerformed(xdata::Event& e)
{

  if (e.type()=="ItemChangedEvent" && fsm_.stateName()->toString()!="Halted") {
    std::string item = dynamic_cast<xdata::ItemChangedEvent&>(e).itemName();
    
    if ( item == "parameterSet") {
      LOG4CPLUS_WARN(getApplicationLogger(),
		     "HLT Menu changed, force reinitialization of EventProcessor");
      epInitialized_ = false;
    }
    if ( item == "outputEnabled") {
      if(outprev_ != outPut_) {
	LOG4CPLUS_WARN(getApplicationLogger(),
		       (outprev_ ? "Disabling " : "Enabling ")<<"global output");
	evtProcessor_->enableEndPaths(outPut_);
	outprev_ = outPut_;
      }
    }
    if (item == "globalInputPrescale") {
      LOG4CPLUS_WARN(getApplicationLogger(),
		     "Setting global input prescale has no effect "
		     <<"in this version of the code");
    }
    if ( item == "globalOutputPrescale") {
      LOG4CPLUS_WARN(getApplicationLogger(),
		     "Setting global output prescale has no effect "
		     <<"in this version of the code");
    }
  }
  
}

//______________________________________________________________________________
void FUEventProcessor::getSlavePids(xgi::Input  *in, xgi::Output *out) throw (xgi::exception::Exception)
{
  for (unsigned int i=0;i<subs_.size();i++)
  {
    if (i!=0) *out << ",";
    *out << subs_[i].pid();
  }
}
//______________________________________________________________________________
void FUEventProcessor::subWeb(xgi::Input  *in, xgi::Output *out) throw (xgi::exception::Exception)
{
  using namespace cgicc;
  pid_t pid = 0;
  std::ostringstream ost;
  ost << "&";

  Cgicc cgi(in);
  internal::MyCgi *mycgi = (internal::MyCgi*)in;
  for(std::map<std::string, std::string, std::less<std::string> >::iterator mit = 
	mycgi->getEnvironment().begin();
      mit != mycgi->getEnvironment().end(); mit++)
    ost << mit->first << "%" << mit->second << ";";
  std::vector<FormEntry> els = cgi.getElements() ;
  std::vector<FormEntry> el1;
  cgi.getElement("method",el1);
  std::vector<FormEntry> el2;
  cgi.getElement("process",el2);
  if(el1.size()!=0) {
    std::string meth = el1[0].getValue();
    if(el2.size()!=0) {
      unsigned int i = 0;
      std::string mod = el2[0].getValue();
      pid = atoi(mod.c_str()); // get the process id to be polled
      for(; i < subs_.size(); i++)
	if(subs_[i].pid()==pid) break;
      if(i>=subs_.size()){ //process was not found, let the browser know
	*out << "ERROR 404 : Process " << pid << " Not Found !" << std::endl;
	return;
      } 
      if(subs_[i].alive() != 1){
	*out << "ERROR 405 : Process " << pid << " Not Alive !" << std::endl;
	return;
      }
      MsgBuf msg1(meth.length()+ost.str().length()+1,MSQM_MESSAGE_TYPE_WEB);
      strncpy(msg1->mtext,meth.c_str(),meth.length());
      strncpy(msg1->mtext+meth.length(),ost.str().c_str(),ost.str().length());
      subs_[i].post(msg1,true);
      unsigned int keep_supersleep_original_value = superSleepSec_.value_;
      superSleepSec_.value_=10*keep_supersleep_original_value;
      MsgBuf msg(MAX_MSG_SIZE,MSQS_MESSAGE_TYPE_WEB);
      bool done = false;
      std::vector<char *>pieces;
      while(!done){
	unsigned long retval1 = subs_[i].rcvNonBlocking(msg,true);
	if(retval1 == MSGQ_MESSAGE_TYPE_RANGE){
	  ::sleep(1);
	  continue;
	}
	unsigned int nbytes = atoi(msg->mtext);
	if(nbytes < MAX_PIPE_BUFFER_SIZE) done = true; // this will break the while loop
	char *buf= new char[nbytes];
	ssize_t retval = read(anonymousPipe_[PIPE_READ],buf,nbytes);
	if(retval<0){
	  std::cout << "Failed to read from pipe." << std::endl;
	  continue;
	}
	if(static_cast<unsigned int>(retval) != nbytes) std::cout 
	  << "CAREFUL HERE, read less bytes than expected from pipe in subWeb" << std::endl;
	pieces.push_back(buf);
      }
      superSleepSec_.value_=keep_supersleep_original_value;
      for(unsigned int j = 0; j < pieces.size(); j++){
	*out<<pieces[j];    // chain the buffers into the output strstream
	delete[] pieces[j]; //make sure to release all buffers used for reading the pipe
      }
    }
  }
}


//______________________________________________________________________________
void FUEventProcessor::defaultWebPage(xgi::Input  *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{


  *out << "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\">" 
       << "<html><head><title>" << getApplicationDescriptor()->getClassName() << (nbSubProcesses_.value_ > 0 ? "MP " : " ") 
       << getApplicationDescriptor()->getInstance() << "</title>"
       << "<meta http-equiv=\"REFRESH\" content=\"0;url=/evf/html/defaultBasePage.html\">"
       << "</head></html>";
}


//______________________________________________________________________________


void FUEventProcessor::spotlightWebPage(xgi::Input  *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{

  std::string urn = getApplicationDescriptor()->getURN();

  *out << "<!-- base href=\"/" <<  urn
       << "\"> -->" << std::endl;
  *out << "<html>"                                                   << std::endl;
  *out << "<head>"                                                   << std::endl;
  *out << "<link type=\"text/css\" rel=\"stylesheet\"";
  *out << " href=\"/evf/html/styles.css\"/>"                   << std::endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName() 
       << getApplicationDescriptor()->getInstance() 
       << " MAIN</title>"     << std::endl;
  *out << "</head>"                                                  << std::endl;
  *out << "<body>"                                                   << std::endl;
  *out << "<table border=\"0\" width=\"100%\">"                      << std::endl;
  *out << "<tr>"                                                     << std::endl;
  *out << "  <td align=\"left\">"                                    << std::endl;
  *out << "    <img"                                                 << std::endl;
  *out << "     align=\"middle\""                                    << std::endl;
  *out << "     src=\"/evf/images/spoticon.jpg\""			     << std::endl;
  *out << "     alt=\"main\""                                        << std::endl;
  *out << "     width=\"64\""                                        << std::endl;
  *out << "     height=\"64\""                                       << std::endl;
  *out << "     border=\"\"/>"                                       << std::endl;
  *out << "    <b>"                                                  << std::endl;
  *out << getApplicationDescriptor()->getClassName() 
       << getApplicationDescriptor()->getInstance()                  << std::endl;
  *out << "      " << fsm_.stateName()->toString()                   << std::endl;
  *out << "    </b>"                                                 << std::endl;
  *out << "  </td>"                                                  << std::endl;
  *out << "  <td width=\"32\">"                                      << std::endl;
  *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << std::endl;
  *out << "      <img"                                               << std::endl;
  *out << "       align=\"middle\""                                  << std::endl;
  *out << "       src=\"/hyperdaq/images/HyperDAQ.jpg\""             << std::endl;
  *out << "       alt=\"HyperDAQ\""                                  << std::endl;
  *out << "       width=\"32\""                                      << std::endl;
  *out << "       height=\"32\""                                     << std::endl;
  *out << "       border=\"\"/>"                                     << std::endl;
  *out << "    </a>"                                                 << std::endl;
  *out << "  </td>"                                                  << std::endl;
  *out << "  <td width=\"32\">"                                      << std::endl;
  *out << "  </td>"                                                  << std::endl;
  *out << "  <td width=\"32\">"                                      << std::endl;
  *out << "    <a href=\"/" << urn << "/\">"                         << std::endl;
  *out << "      <img"                                               << std::endl;
  *out << "       align=\"middle\""                                  << std::endl;
  *out << "       src=\"/evf/images/epicon.jpg\""		     << std::endl;
  *out << "       alt=\"main\""                                      << std::endl;
  *out << "       width=\"32\""                                      << std::endl;
  *out << "       height=\"32\""                                     << std::endl;
  *out << "       border=\"\"/>"                                     << std::endl;
  *out << "    </a>"                                                 << std::endl;
  *out << "  </td>"                                                  << std::endl;
  *out << "</tr>"                                                    << std::endl;
  *out << "</table>"                                                 << std::endl;

  *out << "<hr/>"                                                    << std::endl;
  
  std::ostringstream ost;
  if(myProcess_) 
    ost << "/SubWeb?process=" << getpid() << "&method=moduleWeb&";
  else
    ost << "/moduleWeb?";
  urn += ost.str();
  if(evtProcessor_ && (myProcess_ || nbSubProcesses_.value_==0))
    evtProcessor_.taskWebPage(in,out,urn);
  else if(evtProcessor_)
    evtProcessor_.summaryWebPage(in,out,urn);
  else
    *out << "<td>HLT Unconfigured</td>" << std::endl;
  *out << "</table>"                                                 << std::endl;
  
  *out << "<br><textarea rows=" << 10 << " cols=80 scroll=yes>"      << std::endl;
  *out << configuration_                                             << std::endl;
  *out << "</textarea><P>"                                           << std::endl;
  
  *out << "</body>"                                                  << std::endl;
  *out << "</html>"                                                  << std::endl;


}
void FUEventProcessor::scalersWeb(xgi::Input  *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{

  out->getHTTPResponseHeader().addHeader( "Content-Type",
					  "application/octet-stream" );
  out->getHTTPResponseHeader().addHeader( "Content-Transfer-Encoding",
					  "binary" );
  if(evtProcessor_ != 0){
    out->write( (char*)(evtProcessor_.getPackedTriggerReportAsStruct()), sizeof(TriggerReportStatic));
  }
}

void FUEventProcessor::pathNames(xgi::Input  *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{

  if(evtProcessor_ != 0){
    xdata::Serializable *legenda = scalersLegendaInfoSpace_->find("scalersLegenda");
    if(legenda !=0){
      std::string slegenda = ((xdata::String*)legenda)->value_;
      *out << slegenda << std::endl;
    }
  }
}


void FUEventProcessor::setAttachDqmToShm() throw (evf::Exception)  
{
  std::string errmsg;
  try {
    edm::ServiceRegistry::Operate operate(evtProcessor_->getToken());
    if(edm::Service<FUShmDQMOutputService>().isAvailable())
      edm::Service<FUShmDQMOutputService>()->setAttachToShm();
  }
  catch (cms::Exception& e) {
    errmsg = "Failed to set to attach DQM service to shared memory: " + (std::string)e.what();
  }
  if (!errmsg.empty()) XCEPT_RAISE(evf::Exception,errmsg);
}


void FUEventProcessor::attachDqmToShm() throw (evf::Exception)  
{
  std::string errmsg;
  bool success = false;
  try {
    edm::ServiceRegistry::Operate operate(evtProcessor_->getToken());
    if(edm::Service<FUShmDQMOutputService>().isAvailable())
      success = edm::Service<FUShmDQMOutputService>()->attachToShm();
    if (!success) errmsg = "Failed to attach DQM service to shared memory";
  }
  catch (cms::Exception& e) {
    errmsg = "Failed to attach DQM service to shared memory: " + (std::string)e.what();
  }
  if (!errmsg.empty()) XCEPT_RAISE(evf::Exception,errmsg);
}



void FUEventProcessor::detachDqmFromShm() throw (evf::Exception)
{
  std::string errmsg;
  bool success = false;
  try {
    edm::ServiceRegistry::Operate operate(evtProcessor_->getToken());
    if(edm::Service<FUShmDQMOutputService>().isAvailable())
      success = edm::Service<FUShmDQMOutputService>()->detachFromShm();
    if (!success) errmsg = "Failed to detach DQM service from shared memory";
  }
  catch (cms::Exception& e) {
    errmsg = "Failed to detach DQM service from shared memory: " + (std::string)e.what();
  }
  if (!errmsg.empty()) XCEPT_RAISE(evf::Exception,errmsg);
}


std::string FUEventProcessor::logsAsString()
{
  std::ostringstream oss;
  if(logWrap_)
    {
      for(unsigned int i = logRingIndex_; i < logRing_.size(); i++)
	oss << logRing_[i] << std::endl;
      for(unsigned int i = 0; i <  logRingIndex_; i++)
	oss << logRing_[i] << std::endl;
    }
  else
      for(unsigned int i = logRingIndex_; i < logRing_.size(); i++)
	oss << logRing_[i] << std::endl;
    
  return oss.str();
}
  
void FUEventProcessor::localLog(std::string m)
{
  timeval tv;

  gettimeofday(&tv,0);
  tm *uptm = localtime(&tv.tv_sec);
  char datestring[256];
  strftime(datestring, sizeof(datestring),"%c", uptm);

  if(logRingIndex_ == 0){logWrap_ = true; logRingIndex_ = logRingSize_;}
  logRingIndex_--;
  std::ostringstream timestamp;
  timestamp << " at " << datestring;
  m += timestamp.str();
  logRing_[logRingIndex_] = m;
}

void FUEventProcessor::startSupervisorLoop()
{
  try {
    wlSupervising_=
      toolbox::task::getWorkLoopFactory()->getWorkLoop("Supervisor",
						       "waiting");
    if (!wlSupervising_->isActive()) wlSupervising_->activate();
    asSupervisor_ = toolbox::task::bind(this,&FUEventProcessor::supervisor,
					"Supervisor");
    wlSupervising_->submit(asSupervisor_);
    supervising_ = true;
  }
  catch (xcept::Exception& e) {
    std::string msg = "Failed to start workloop 'Supervisor'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
}

void FUEventProcessor::startReceivingLoop()
{
  try {
    wlReceiving_=
      toolbox::task::getWorkLoopFactory()->getWorkLoop("Receiving",
						       "waiting");
    if (!wlReceiving_->isActive()) wlReceiving_->activate();
    asReceiveMsgAndExecute_ = toolbox::task::bind(this,&FUEventProcessor::receiving,
					"Receiving");
    wlReceiving_->submit(asReceiveMsgAndExecute_);
    receiving_ = true;
  }
  catch (xcept::Exception& e) {
    std::string msg = "Failed to start workloop 'Receiving'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
}
void FUEventProcessor::startReceivingMonitorLoop()
{
  try {
    wlReceivingMonitor_=
      toolbox::task::getWorkLoopFactory()->getWorkLoop("ReceivingM",
						       "waiting");
    if (!wlReceivingMonitor_->isActive()) 
      wlReceivingMonitor_->activate();
    asReceiveMsgAndRead_ = 
      toolbox::task::bind(this,&FUEventProcessor::receivingAndMonitor,
			  "ReceivingM");
    wlReceivingMonitor_->submit(asReceiveMsgAndRead_);
    receivingM_ = true;
  }
  catch (xcept::Exception& e) {
    std::string msg = "Failed to start workloop 'ReceivingM'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
}

bool FUEventProcessor::receiving(toolbox::task::WorkLoop *)
{
  MsgBuf msg;
  try{
    myProcess_->rcvSlave(msg,false); //will receive only messages from Master
    if(msg->mtype==MSQM_MESSAGE_TYPE_RLI)
      {
	rlimit rl;
	getrlimit(RLIMIT_CORE,&rl);
	rl.rlim_cur=0;
	setrlimit(RLIMIT_CORE,&rl);
	rlimit_coresize_changed_=true;
      }
    if (msg->mtype==MSQM_MESSAGE_TYPE_RLR)
      {
	//reset coresize limit
        setrlimit(RLIMIT_CORE,&rlimit_coresize_default_);
	rlimit_coresize_changed_=false;
      }
    if(msg->mtype==MSQM_MESSAGE_TYPE_STOP)
      {
	pthread_mutex_lock(&stop_lock_);
	try {
	  fsm_.fireEvent("Stop",this); // need to set state in fsm first to allow stopDone transition
	}
	catch (...) {
	  LOG4CPLUS_ERROR(getApplicationLogger(),"Failed to go to Stopping state in slave EP, pid "
			                         << getpid() << " The state on Stop event was not consistent");
	}

	try {
	  stopClassic(); // call the normal sequence of stopping - as this is allowed to fail provisions must be made ...@@@EM
	}
	catch (...) {
	  LOG4CPLUS_ERROR(getApplicationLogger(),"Slave EP 'receiving' workloop: exception " << getpid());
	}

        //destroy MessageService thread before exit	
	try{
	  messageServicePresence_.reset();
	}
	catch(...) {
	  LOG4CPLUS_ERROR(getApplicationLogger(),"SLAVE:Unable to destroy MessageServicePresence. pid:" << getpid() );
	}

	MsgBuf msg1(0,MSQS_MESSAGE_TYPE_STOP);
	myProcess_->postSlave(msg1,false);
	pthread_mutex_unlock(&stop_lock_);
	fclose(stdout);
	fclose(stderr);
	_exit(EXIT_SUCCESS);
      }
    if(msg->mtype==MSQM_MESSAGE_TYPE_FSTOP)
      _exit(EXIT_SUCCESS);
  }
  catch(evf::Exception &e){
    LOG4CPLUS_ERROR(getApplicationLogger(),"Slave EP pid:" << getpid() << " receiving WorkLoop exception: "<<e.what());
  }
  return true;
}

bool FUEventProcessor::supervisor(toolbox::task::WorkLoop *)
{
  pthread_mutex_lock(&stop_lock_);
  if(subs_.size()!=nbSubProcesses_.value_)
    {
      pthread_mutex_lock(&pickup_lock_);
      if(subs_.size()!=nbSubProcesses_.value_) {
        subs_.resize(nbSubProcesses_.value_);
        spMStates_.resize(nbSubProcesses_.value_);
        spmStates_.resize(nbSubProcesses_.value_);
        for(unsigned int i = 0; i < spMStates_.size(); i++)
	  {
	    spMStates_[i] = edm::event_processor::sInit; 
	    spmStates_[i] = 0; 
	  }
      }
      pthread_mutex_unlock(&pickup_lock_);
    }
  bool running = fsm_.stateName()->toString()=="Enabled";
  bool stopping = fsm_.stateName()->toString()=="stopping";
  for(unsigned int i = 0; i < subs_.size(); i++)
    {
      if(subs_[i].alive()==-1000) continue;
      int sl;
      pid_t sub_pid = subs_[i].pid();
      pid_t killedOrNot = waitpid(sub_pid,&sl,WNOHANG);

      if(killedOrNot && killedOrNot==sub_pid) {
	pthread_mutex_lock(&pickup_lock_);
	//check if out of range or recreated (enable can clear vector)
	if (i<subs_.size() && subs_[i].alive()!=-1000) {
	  subs_[i].setStatus((WIFEXITED(sl) != 0 ? 0 : -1));
	  std::ostringstream ost;
	  if(subs_[i].alive()==0) ost << " process exited with status " << WEXITSTATUS(sl);
	  else if(WIFSIGNALED(sl)!=0) {
	    ost << " process terminated with signal " << WTERMSIG(sl);
	  }
	  else ost << " process stopped ";
	  //report unexpected slave exit in stop
	  //if (stopping && (WEXITSTATUS(sl)!=0 || WIFSIGNALED(sl)!=0)) {
	  //  LOG4CPLUS_WARN(getApplicationLogger(),ost.str() << ", slave pid:"<<getpid());
	  //}
	  subs_[i].countdown()=slaveRestartDelaySecs_.value_;
	  subs_[i].setReasonForFailed(ost.str());
	  spMStates_[i] = evtProcessor_.notstarted_state_code();
	  spmStates_[i] = 0;
	  std::ostringstream ost1;
	  ost1 << "-E- Slave " << subs_[i].pid() << ost.str();
	  localLog(ost1.str());
	  if(!autoRestartSlaves_.value_) subs_[i].disconnect();
	}
	pthread_mutex_unlock(&pickup_lock_);
      }
    }
  pthread_mutex_unlock(&stop_lock_);	
  if(stopping) return true; // if in stopping we are done

  // check if we need to reset core dumps (15 min after last one)
  if (running && rlimit_coresize_changed_) {
    timeval newtv;
    gettimeofday(&newtv,0);
    int delta = newtv.tv_sec-lastCrashTime_.tv_sec;
    if (delta>60*15) {
      std::ostringstream ostr;
      ostr << " No more slave EP crashes on this machine in last 15 min. resetting core size limits";
      std::cout << ostr.str() << std::endl;
      LOG4CPLUS_INFO(getApplicationLogger(),ostr.str());
      setrlimit(RLIMIT_CORE,&rlimit_coresize_default_);
      MsgBuf master_message_rlr_(NUMERIC_MESSAGE_SIZE,MSQM_MESSAGE_TYPE_RLR);
      for (unsigned int i = 0; i < subs_.size(); i++) {
	try {
	  if (subs_[i].alive())
	    subs_[i].post(master_message_rlr_,false);
	}
	catch (...) {}
      }
      rlimit_coresize_changed_=false;
      crashesThisRun_=0;
    }
  }

  if(running && edm_init_done_)
    {
      // if enabled, this loop will periodically check if dead slaves countdown has expired and restart them
      // this is only active while running, hence, the stop lock is acquired and only released at end of loop
      if(autoRestartSlaves_.value_){
	pthread_mutex_lock(&stop_lock_); //lockout slave killing at stop while you check for restarts
	for(unsigned int i = 0; i < subs_.size(); i++)
	  {
	    if(subs_[i].alive() != 1){
	      if(subs_[i].countdown() == 0)
		{
		  if(subs_[i].restartCount()>2){
		    LOG4CPLUS_WARN(getApplicationLogger()," Not restarting subprocess in slot " << i 
				   << " - maximum restart count reached");
		    std::ostringstream ost1;
		    ost1 << "-W- Dead Process in slot " << i << " reached maximum restart count"; 
		    localLog(ost1.str());
		    subs_[i].countdown()--;
		    XCEPT_DECLARE(evf::Exception,
				  sentinelException, ost1.str());
		    notifyQualified("error",sentinelException);
		    subs_[i].disconnect();
		    continue;
		  }
		  subs_[i].restartCount()++;
		  if (forkInEDM_.value_) {
		    restartForkInEDM(i);
		  }
		  else {
		    pid_t rr = subs_[i].forkNew();
		    if(rr==0)
		    {
		      myProcess_=&subs_[i];
		      scalersUpdates_ = 0;
		      int retval = pthread_mutex_destroy(&stop_lock_);
		      if(retval != 0) perror("error");
		      retval = pthread_mutex_init(&stop_lock_,0);
		      if(retval != 0) perror("error");
		      fsm_.disableRcmsStateNotification();
		      fsm_.fireEvent("Stop",this); // need to set state in fsm first to allow stopDone transition
		      fsm_.fireEvent("StopDone",this); // need to set state in fsm first to allow stopDone transition
		      fsm_.fireEvent("Enable",this); // need to set state in fsm first to allow stopDone transition
		      try{
			xdata::Serializable *lsid = applicationInfoSpace_->find("lumiSectionIndex");
			if(lsid) {
			  ((xdata::UnsignedInteger32*)(lsid))->value_--; // need to reset to value before end of ls in which process died
			}
		      }
		      catch(...){
			std::cout << "trouble with lsindex during restart" << std::endl;
		      }
		      try{
			xdata::Serializable *lstb = applicationInfoSpace_->find("lsToBeRecovered");
			if(lstb) {
			  ((xdata::Boolean*)(lstb))->value_ = false; // do not issue eol/bol for all Ls when restarting
			}
		      }
		      catch(...){
			std::cout << "trouble with resetting flag for eol recovery " << std::endl;
		      }

		      evtProcessor_.adjustLsIndexForRestart();
		      evtProcessor_.resetPackedTriggerReport();
		      enableMPEPSlave();
		      return false; // exit the supervisor loop immediately in the child !!!
		    }
		  else
		    {
		      std::ostringstream ost1;
		      ost1 << "-I- New Process " << rr << " forked for slot " << i; 
		      localLog(ost1.str());
		    }
		  }
		}
	      if(subs_[i].countdown()>=0) subs_[i].countdown()--;
	    }
	  }
	pthread_mutex_unlock(&stop_lock_);
      } // finished handling replacement of dead slaves once they've been reaped
    }
  xdata::Serializable *lsid = 0; 
  xdata::Serializable *psid = 0;
  xdata::Serializable *dqmp = 0;
  xdata::UnsignedInteger32 *dqm = 0;


  
  if(running && edm_init_done_){  
    try{
      lsid = applicationInfoSpace_->find("lumiSectionIndex");
      psid = applicationInfoSpace_->find("prescaleSetIndex");
      nbProcessed = monitorInfoSpace_->find("nbProcessed");
      nbAccepted  = monitorInfoSpace_->find("nbAccepted");
      dqmp = applicationInfoSpace_-> find("nbDqmUpdates");      
    }
    catch(xdata::exception::Exception e){
      LOG4CPLUS_INFO(getApplicationLogger(),"could not retrieve some data - " << e.what());    
    }

    try{
      if(nbProcessed !=0 && nbAccepted !=0)
	{
	  xdata::UnsignedInteger32*nbp = ((xdata::UnsignedInteger32*)nbProcessed);
	  xdata::UnsignedInteger32*nba = ((xdata::UnsignedInteger32*)nbAccepted);
	  xdata::UnsignedInteger32*ls  = ((xdata::UnsignedInteger32*)lsid);
	  xdata::UnsignedInteger32*ps  = ((xdata::UnsignedInteger32*)psid);
	  if(dqmp!=0)
	    dqm = (xdata::UnsignedInteger32*)dqmp;
	  if(dqm) dqm->value_ = 0;
	  nbTotalDQM_ = 0;
	  nbp->value_ = 0;
	  nba->value_ = 0;
	  nblive_ = 0;
	  nbdead_ = 0;
	  scalersUpdates_ = 0;

	  for(unsigned int i = 0; i < subs_.size(); i++)
	    {
	      if(subs_[i].alive()>0)
		{
		  nblive_++;
		  try{
		    subs_[i].post(master_message_prg_,true);
		    
		    unsigned long retval = subs_[i].rcvNonBlocking(master_message_prr_,true);
		    if(retval == (unsigned long) master_message_prr_->mtype){
		      prg* p = (struct prg*)(master_message_prr_->mtext);
		      subs_[i].setParams(p);
		      spMStates_[i] = p->Ms;
		      spmStates_[i] = p->ms;
		      cpustat_->addEntry(p->ms);
		      if(!subs_[i].inInconsistentState() && 
			 (p->Ms == edm::event_processor::sError 
			  || p->Ms == edm::event_processor::sInvalid
			  || p->Ms == edm::event_processor::sStopping))
			{
			  std::ostringstream ost;
			  ost << "edm::eventprocessor slot " << i << " process id " 
			      << subs_[i].pid() << " not in Running state : Mstate=" 
			      << evtProcessor_.stateNameFromIndex(p->Ms) << " mstate="
			      << evtProcessor_.moduleNameFromIndex(p->ms) 
			      << " - Look into possible error messages from HLT process";
			  LOG4CPLUS_WARN(getApplicationLogger(),ost.str());
			}
		      nbp->value_ += subs_[i].params().nbp;
		      nba->value_  += subs_[i].params().nba;
		      if(dqm)dqm->value_ += p->dqm;
		      nbTotalDQM_ +=  p->dqm;
		      scalersUpdates_ += p->trp;
		      if(p->ls > ls->value_) ls->value_ = p->ls;
		      if(p->ps != ps->value_) ps->value_ = p->ps;
		    }
		    else{
		      nbp->value_ += subs_[i].get_save_nbp();
		      nba->value_ += subs_[i].get_save_nba();
		    }
		  } 
		  catch(evf::Exception &e){
		    LOG4CPLUS_INFO(getApplicationLogger(),
				   "could not send/receive msg on slot " 
				   << i << " - " << e.what());    
		  }
		    
		}
	      else
		{
		  nbp->value_ += subs_[i].get_save_nbp();
		  nba->value_ += subs_[i].get_save_nba();
		  nbdead_++;
		}
	    }
	  if(nbp->value_>64){//have some slaves already processed more than one event ? (eventually make this == number of raw cells)
	    for(unsigned int i = 0; i < subs_.size(); i++)
	      {
		if(subs_[i].params().nbp == 0){ // a slave has processed 0 events 
		  // check that the process is not stuck
		  if(subs_[i].alive()>0 && subs_[i].params().ms == 0) // the process is seen alive but in us=Invalid(0)
		    {
		      subs_[i].found_invalid();//increase the "found_invalid" counter
		      if(subs_[i].nfound_invalid() > 60){ //wait x monitor cycles (~1 min a good time ?) before doing something about a stuck slave
			MsgBuf msg3(NUMERIC_MESSAGE_SIZE,MSQM_MESSAGE_TYPE_FSTOP);	// send a force-stop signal		
			subs_[i].post(msg3,false);
			std::ostringstream ost1;
			ost1 << "-W- Process in slot " << i << " Never reached the running state - forcestopping it"; 
			localLog(ost1.str());
			LOG4CPLUS_ERROR(getApplicationLogger(),ost1.str());    
			XCEPT_DECLARE(evf::Exception,
				      sentinelException, ost1.str());
			notifyQualified("error",sentinelException);

		      }
		    }
		}
	      }
	  }
	}
    }
    catch(std::exception &e){
      LOG4CPLUS_INFO(getApplicationLogger(),"std exception - " << e.what());    
    }
    catch(...){
      LOG4CPLUS_INFO(getApplicationLogger(),"unknown exception ");    
    }
  }
  else{
    for(unsigned int i = 0; i < subs_.size(); i++)
      {
	if(subs_[i].alive()==-1000)
	  {
	    spMStates_[i] = edm::event_processor::sInit;
	    spmStates_[i] = 0;
	  }
      }
  }
  try{
    monitorInfoSpace_->lock();
    monitorInfoSpace_->fireItemGroupChanged(names_,0);
    monitorInfoSpace_->unlock();
  }
  catch(xdata::exception::Exception &e)
    {
      LOG4CPLUS_ERROR(log_, "Exception from fireItemGroupChanged: " << e.what());
      //	localLog(e.what());
    }
  ::sleep(superSleepSec_.value_);	
  return true;
}

void FUEventProcessor::startScalersWorkLoop() throw (evf::Exception)
{
  try {
    wlScalers_=
      toolbox::task::getWorkLoopFactory()->getWorkLoop("Scalers",
						       "waiting");
    if (!wlScalers_->isActive()) wlScalers_->activate();
    asScalers_ = toolbox::task::bind(this,&FUEventProcessor::scalers,
				     "Scalers");
    
  wlScalers_->submit(asScalers_);
  wlScalersActive_ = true;
  }
  catch (xcept::Exception& e) {
    std::string msg = "Failed to start workloop 'Scalers'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
}

//______________________________________________________________________________

void FUEventProcessor::startSummarizeWorkLoop() throw (evf::Exception)
{
  try {
    wlSummarize_=
      toolbox::task::getWorkLoopFactory()->getWorkLoop("Summary",
						       "waiting");
    if (!wlSummarize_->isActive()) wlSummarize_->activate();
    
    asSummarize_ = toolbox::task::bind(this,&FUEventProcessor::summarize,
				       "Summary");

    wlSummarize_->submit(asSummarize_);
    wlSummarizeActive_ = true;
  }
  catch (xcept::Exception& e) {
    std::string msg = "Failed to start workloop 'Summarize'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
}

//______________________________________________________________________________

bool FUEventProcessor::scalers(toolbox::task::WorkLoop* wl)
{
  if(evtProcessor_)
    {
      if(!evtProcessor_.getTriggerReport(true)) {
	wlScalersActive_ = false;
	return false;
      }
    }
  else
    {
      std::cout << getpid()<< " Scalers workloop, bailing out, no evtProcessor " << std::endl;
      wlScalersActive_ = false;
      return false;
    }
  if(myProcess_) 
    {
      //      std::cout << getpid() << "going to post on control queue from scalers" << std::endl;
      int ret = myProcess_->postSlave(evtProcessor_.getPackedTriggerReport(),false);
      if(ret!=0)      std::cout << "scalers workloop, error posting to sqs_ " << errno << std::endl;
      scalersUpdates_++;
    }
  else
    evtProcessor_.fireScalersUpdate();
  return true;
}

//______________________________________________________________________________
bool FUEventProcessor::summarize(toolbox::task::WorkLoop* wl)
{
  evtProcessor_.resetPackedTriggerReport();
  bool atLeastOneProcessUpdatedSuccessfully = false;
  int msgCount = 0;
  for (unsigned int i = 0; i < subs_.size(); i++)
    {
      if(subs_[i].alive()>0)
	{
	  int ret = 0;
	  if(subs_[i].check_postponed_trigger_update(master_message_trr_,
						     evtProcessor_.getLumiSectionReferenceIndex()))
	    {
	      ret = MSQS_MESSAGE_TYPE_TRR;
	      std::cout << "using postponed report from slot " << i << " for ls " << evtProcessor_.getLumiSectionReferenceIndex() << std::endl;
	    }
	  else{
	    bool insync = false;
	    bool exception_caught = false;
	    while(!insync){
	      try{
		ret = subs_[i].rcv(master_message_trr_,false);
	      }
	      catch(evf::Exception &e)
		{
		  std::cout << "exception in msgrcv on " << i 
			    << " " << subs_[i].alive() << " " << strerror(errno) << std::endl;
		  exception_caught = true;
		  break;
		  //do nothing special
		}
	      if(ret==MSQS_MESSAGE_TYPE_TRR) {
		TriggerReportStatic *trp = (TriggerReportStatic *)master_message_trr_->mtext;
		if(trp->lumiSection >= evtProcessor_.getLumiSectionReferenceIndex()){
		  insync = true;
		}
	      }
	    }
	    if(exception_caught) continue;
	  }
	  msgCount++;
	  if(ret==MSQS_MESSAGE_TYPE_TRR) {
	    TriggerReportStatic *trp = (TriggerReportStatic *)master_message_trr_->mtext;
	    if(trp->lumiSection > evtProcessor_.getLumiSectionReferenceIndex()){
	      std::cout << "postpone handling of msg from slot " << i << " with Ls " <<  trp->lumiSection
			<< " should be " << evtProcessor_.getLumiSectionReferenceIndex() << std::endl;
	      subs_[i].add_postponed_trigger_update(master_message_trr_);
	    }else{
	      atLeastOneProcessUpdatedSuccessfully = true;
	      evtProcessor_.sumAndPackTriggerReport(master_message_trr_);
	    }
	  }
	  else std::cout << "msgrcv returned error " << errno << std::endl;
	}
    }
  if(atLeastOneProcessUpdatedSuccessfully){
    nbSubProcessesReporting_.value_ = msgCount;
    evtProcessor_.getPackedTriggerReportAsStruct()->nbExpected = nbSubProcesses_.value_;
    evtProcessor_.getPackedTriggerReportAsStruct()->nbReporting = nbSubProcessesReporting_.value_;
    evtProcessor_.updateRollingReport();
    evtProcessor_.fireScalersUpdate();
  }
  else{
    LOG4CPLUS_WARN(getApplicationLogger(),"Summarize loop: no process updated successfully - sleep 10 seconds before trying again");          
    if(msgCount==0) evtProcessor_.withdrawLumiSectionIncrement();
    nbSubProcessesReporting_.value_ = 0;
    ::sleep(10);
  }
  if(fsm_.stateName()->toString()!="Enabled"){
    wlScalersActive_ = false;
    return false;
  }
  //  cpustat_->printStat();
  if(iDieStatisticsGathering_.value_){
    try{
      unsigned long long idleTmp=idleProcStats_;
      unsigned long long allPSTmp=allProcStats_;
      idleProcStats_=allProcStats_=0;

      utils::procCpuStat(idleProcStats_,allProcStats_);
      timeval oldtime=lastProcReport_;
      gettimeofday(&lastProcReport_,0);

      if (allPSTmp!=0 && idleTmp!=0 && allProcStats_!=allPSTmp) {
	cpustat_->setCPUStat(1000 - ((idleProcStats_-idleTmp)*1000)/(allProcStats_-allPSTmp));
        int deltaTms=1000 * (lastProcReport_.tv_sec-oldtime.tv_sec)
	            + (lastProcReport_.tv_usec-oldtime.tv_usec)/1000;
	cpustat_->setElapsed(deltaTms);
      }
      else {
	cpustat_->setCPUStat(0);
        cpustat_->setElapsed(0);
      }

      TriggerReportStatic *trsp = evtProcessor_.getPackedTriggerReportAsStruct();
      cpustat_ ->setNproc(trsp->eventSummary.totalEvents);
      cpustat_ ->sendStat(evtProcessor_.getLumiSectionReferenceIndex());
      ratestat_->sendStat((unsigned char*)trsp,
			  sizeof(TriggerReportStatic),
			  evtProcessor_.getLumiSectionReferenceIndex());
    }catch(evf::Exception &e){
      LOG4CPLUS_INFO(getApplicationLogger(),"coud not send statistics"
		     << e.what());
    }
  }
  cpustat_->reset();
  return true;
}



bool FUEventProcessor::receivingAndMonitor(toolbox::task::WorkLoop *)
{
  try{
    myProcess_->rcvSlave(slave_message_monitoring_,true); //will receive only messages from Master
    switch(slave_message_monitoring_->mtype)
      {
      case MSQM_MESSAGE_TYPE_MCS:
	{
	  xgi::Input *in = 0;
	  xgi::Output out;
	  evtProcessor_.microState(in,&out);
	  MsgBuf msg1(out.str().size(),MSQS_MESSAGE_TYPE_MCR);
	  strncpy(msg1->mtext,out.str().c_str(),out.str().size());
	  myProcess_->postSlave(msg1,true);
	  break;
	}
      
      case MSQM_MESSAGE_TYPE_PRG:
	{
	  xdata::Serializable *dqmp = 0;
	  xdata::UnsignedInteger32 *dqm = 0;
	  evtProcessor_.monitoring(0);
	  try{
	    dqmp = applicationInfoSpace_-> find("nbDqmUpdates");
	  }  catch(xdata::exception::Exception e){}
	  if(dqmp!=0)
	    dqm = (xdata::UnsignedInteger32*)dqmp;

	  // 	  monitorInfoSpace_->lock();  
	  prg * data           = (prg*)slave_message_prr_->mtext;
	  data->ls             = evtProcessor_.lsid_;
	  data->eols           = evtProcessor_.lastLumiUsingEol_;
	  data->ps             = evtProcessor_.psid_;
	  data->nbp            = evtProcessor_->totalEvents();
	  data->nba            = evtProcessor_->totalEventsPassed();
	  data->Ms             = evtProcessor_.epMAltState_.value_;
	  data->ms             = evtProcessor_.epmAltState_.value_;
	  if(dqm) data->dqm    = dqm->value_; else data->dqm = 0;
	  data->trp            = scalersUpdates_;
	  //	  monitorInfoSpace_->unlock();  
	  myProcess_->postSlave(slave_message_prr_,true);
	  if(exitOnError_.value_)
	  { 
	    // after each monitoring cycle check if we are in inconsistent state and exit if configured to do so  
	    //	    std::cout << getpid() << "receivingAndMonitor: trying to acquire stop lock " << std::endl;
	    if(data->Ms == edm::event_processor::sError) 
	      { 
		bool running = true;
		int count = 0;
		while(running){
		  int retval = pthread_mutex_lock(&stop_lock_);
		  if(retval != 0) perror("error");
		  running = fsm_.stateName()->toString()=="Enabled";
		  if(count>5) _exit(-1);
		  pthread_mutex_unlock(&stop_lock_);
		  if(running) {::sleep(1); count++;}
		}
	      }
	  }
	  break;
	}
      case MSQM_MESSAGE_TYPE_WEB:
	{
	  xgi::Input  *in = 0;
	  xgi::Output out;
	  unsigned int bytesToSend = 0;
	  MsgBuf msg1(NUMERIC_MESSAGE_SIZE,MSQS_MESSAGE_TYPE_WEB);
	  std::string query = slave_message_monitoring_->mtext;
	  size_t pos = query.find_first_of("&");
	  std::string method;
	  std::string args;
	  if(pos!=std::string::npos)  
	    {
	      method = query.substr(0,pos);
	      args = query.substr(pos+1,query.length()-pos-1);
	    }
	  else
	    method=query;

	  if(method=="Spotlight")
	    {
	      spotlightWebPage(in,&out);
	    }
	  else if(method=="procStat")
	    {
	      procStat(in,&out);
	    }
	  else if(method=="moduleWeb")
	    {
	      internal::MyCgi mycgi;
	      boost::char_separator<char> sep(";");
	      boost::tokenizer<boost::char_separator<char> > tokens(args, sep);
	      for (boost::tokenizer<boost::char_separator<char> >::iterator tok_iter = tokens.begin();
		   tok_iter != tokens.end(); ++tok_iter){
		size_t pos = (*tok_iter).find_first_of("%");
		if(pos != std::string::npos){
		  std::string first  = (*tok_iter).substr(0    ,                        pos);
		  std::string second = (*tok_iter).substr(pos+1, (*tok_iter).length()-pos-1);
		  mycgi.getEnvironment()[first]=second;
		}
	      }
	      moduleWeb(&mycgi,&out);
	    }
	  else if(method=="Default")
	    {
	      defaultWebPage(in,&out);
	    }
	  else 
	    {
	      out << "Error 404!!!!!!!!" << std::endl;
	    }


	  bytesToSend = out.str().size();
	  unsigned int cycle = 0;
	  if(bytesToSend==0)
	    {
	      snprintf(msg1->mtext, NUMERIC_MESSAGE_SIZE, "%d", bytesToSend);
	      myProcess_->postSlave(msg1,true);
	    }
	  while(bytesToSend !=0){
	    unsigned int msgSize = bytesToSend>MAX_PIPE_BUFFER_SIZE ? MAX_PIPE_BUFFER_SIZE : bytesToSend;
	    write(anonymousPipe_[PIPE_WRITE],
		  out.str().c_str()+MAX_PIPE_BUFFER_SIZE*cycle,
		  msgSize);
	    snprintf(msg1->mtext, NUMERIC_MESSAGE_SIZE, "%d", msgSize);
	    myProcess_->postSlave(msg1,true);
	    bytesToSend -= msgSize;
	    cycle++;
	  }
	  break;
	}
      case MSQM_MESSAGE_TYPE_TRP:
	{
	  break;
	}
      }
  }
  catch(evf::Exception &e){std::cout << "exception caught in recevingM: " << e.what() << std::endl;}
  return true;
}

void FUEventProcessor::startSignalMonitorWorkLoop() throw (evf::Exception)
{
  //todo rewind/check semaphore
  //start workloop
  try {
    wlSignalMonitor_=
      toolbox::task::getWorkLoopFactory()->getWorkLoop("SignalMonitor",
						       "waiting");

    if (!wlSignalMonitor_->isActive()) wlSignalMonitor_->activate();
    asSignalMonitor_ = toolbox::task::bind(this,&FUEventProcessor::sigmon,
				       "SignalMonitor");
    wlSignalMonitor_->submit(asSignalMonitor_);
    signalMonitorActive_ = true;
  }
  catch (xcept::Exception& e) {
    std::string msg = "Failed to start workloop 'SignalMonitor'. (3)";
    std::cout << e.what() << std::endl;
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
}
 

bool FUEventProcessor::sigmon(toolbox::task::WorkLoop* wl)
{
  while (1) {
    sem_wait(sigmon_sem_);
    std::cout << " received signal notification from slave!"<< std::endl;
  
    //check if shutdown time
    bool running = fsm_.stateName()->toString()=="Enabled";
    bool stopping = fsm_.stateName()->toString()=="stopping";
    bool enabling = fsm_.stateName()->toString()=="enabling";
    if (!running && !enabling) {
      signalMonitorActive_ = false;
      return false;
    }

    crashesThisRun_++;
    gettimeofday(&lastCrashTime_,0);

    //set core size limit to 0 in master and slaves
    if (crashesThisRun_>=crashesToDump_.value_ && (running || stopping) && !rlimit_coresize_changed_) {

      rlimit rlold;
      getrlimit(RLIMIT_CORE,&rlold);
      rlimit rlnew = rlold;
      rlnew.rlim_cur=0;
      setrlimit(RLIMIT_CORE,&rlnew);
      rlimit_coresize_changed_=true;
      MsgBuf master_message_rli_(NUMERIC_MESSAGE_SIZE,MSQM_MESSAGE_TYPE_RLI);
      //in case of frequent crashes, allow first slot to dump (until restart)
      unsigned int min=1;
      for (unsigned int i = min; i < subs_.size(); i++) {
	try {
	  if (subs_[i].alive()) {
	    subs_[i].post(master_message_rli_,false);
	  }
	}
	catch (...) {}
      }
      std::ostringstream ostr;
      ostr << "Number of recent slave crashes reaches " << crashesThisRun_
           << ". Disabling core dumps for next 15 minutes in this FilterUnit";
      LOG4CPLUS_WARN(getApplicationLogger(),ostr.str());
    }
  }//end while loop
  signalMonitorActive_ = false;
  return false;
}


bool FUEventProcessor::enableCommon()
{
  try {    
    if(hasShMem_) attachDqmToShm();
    int sc = 0;
    evtProcessor_->clearCounters();
    if(isRunNumberSetter_)
      evtProcessor_->setRunNumber(runNumber_.value_);
    else
      evtProcessor_->declareRunNumber(runNumber_.value_);

    try{
      ::sleep(1);
      evtProcessor_->runAsync();
      sc = evtProcessor_->statusAsync();
    }
    catch(cms::Exception &e) {
      reasonForFailedState_ = e.explainSelf();
      fsm_.fireFailed(reasonForFailedState_,this);
      localLog(reasonForFailedState_);
      return false;
    }    
    catch(std::exception &e) {
      reasonForFailedState_  = e.what();
      fsm_.fireFailed(reasonForFailedState_,this);
      localLog(reasonForFailedState_);
      return false;
    }
    catch(...) {
      reasonForFailedState_ = "Unknown Exception";
      fsm_.fireFailed(reasonForFailedState_,this);
      localLog(reasonForFailedState_);
      return false;
    }
    if(sc != 0) {
      std::ostringstream oss;
      oss<<"EventProcessor::runAsync returned status code " << sc;
      reasonForFailedState_ = oss.str();
      fsm_.fireFailed(reasonForFailedState_,this);
      localLog(reasonForFailedState_);
      return false;
    }
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "enabling FAILED: " + (std::string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    localLog(reasonForFailedState_);
    return false;
  }
  try{
    fsm_.fireEvent("EnableDone",this);
  }
  catch (xcept::Exception &e) {
    std::cout << "exception " << (std::string)e.what() << std::endl;
    throw;
  }

  return false;
}

void FUEventProcessor::forkProcessFromEDM_helper(void * addr) {
  ((FUEventProcessor*)addr)->forkProcessesFromEDM();
}

void FUEventProcessor::forkProcessesFromEDM() {

  moduleweb::ForkParams * forkParams = &(forkInfoObj_->forkParams);
  unsigned int forkFrom=0;
  unsigned int forkTo=nbSubProcesses_.value_;
  if (forkParams->slotId>=0) {
    forkFrom=forkParams->slotId;
    forkTo=forkParams->slotId+1;
  }

  //before fork, make sure to disconnect output modules from Shm
  try {
    if (sorRef_) {
      std::vector<edm::FUShmOutputModule *> shmOutputs = sorRef_->getShmOutputModules();
      for (unsigned int i=0;i<shmOutputs.size();i++) {
        //unregister PID from ShmBuffer/RB
        shmOutputs[i]->unregisterFromShm();
	//disconnect from Shm
        shmOutputs[i]->stop();
      }
    }
  }
  catch (std::exception &e)
  {
    reasonForFailedState_ =  (std::string)"Thrown exception while disconnecting ShmOutputModule from Shm: " + e.what();
    LOG4CPLUS_ERROR(getApplicationLogger(),reasonForFailedState_);
    fsm_.fireFailed(reasonForFailedState_,this);
    localLog(reasonForFailedState_);
  }
  catch (...) {
    reasonForFailedState_ =  "Thrown unknown exception while disconnecting ShmOutputModule from Shm: ";
    LOG4CPLUS_ERROR(getApplicationLogger(),reasonForFailedState_);
    fsm_.fireFailed(reasonForFailedState_,this);
    localLog(reasonForFailedState_);
  }

  std::string currentState = fsm_.stateName()->toString();

  //destroy MessageServicePresence thread before fork
  if (currentState!="stopping") {
    try {
      messageServicePresence_.reset();
    }
    catch (...) {
      LOG4CPLUS_ERROR(getApplicationLogger(),"Unable to destroy MessageService thread before fork!");
    }
  }

  if (currentState=="stopping") {
    LOG4CPLUS_ERROR(getApplicationLogger(),"Can not fork subprocesses in state " << fsm_.stateName()->toString());
    forkParams->isMaster=1;
    forkInfoObj_->forkParams.slotId=-1;
    forkInfoObj_->forkParams.restart=0;
  }
  //fork loop
  else for(unsigned int i=forkFrom; i<forkTo; i++)
  {
    int retval = subs_[i].forkNew();
    if(retval==0)
    {
      forkParams->isMaster=0;
      myProcess_ = &subs_[i];
      // dirty hack: delete/recreate global binary semaphore for later use in child
      delete toolbox::mem::_s_mutex_ptr_;
      toolbox::mem::_s_mutex_ptr_ = new toolbox::BSem(toolbox::BSem::FULL,true);
      int retval = pthread_mutex_destroy(&stop_lock_);
      if(retval != 0) perror("error");
      retval = pthread_mutex_init(&stop_lock_,0);
      if(retval != 0) perror("error");
      fsm_.disableRcmsStateNotification();

      //recreate MessageLogger thread in slave after fork
      try{
	edm::PresenceFactory *pf = edm::PresenceFactory::get();
	if(pf != 0) {
	  messageServicePresence_ = pf->makePresence("MessageServicePresence");
	}
	else {
	  LOG4CPLUS_ERROR(getApplicationLogger(),
	      "SLAVE: Unable to create message service presence. pid:"<<getpid());
	}
      }
      catch(...) {
        LOG4CPLUS_ERROR(getApplicationLogger(),"SLAVE: Unknown Exception in MessageServicePresence. pid:"<<getpid());
      }

      ML::MLlog4cplus::setAppl(this);

      //reconnect to Shm from output modules
      try {
        if (sorRef_) {
	  std::vector<edm::FUShmOutputModule *> shmOutputs = sorRef_->getShmOutputModules();
	  for (unsigned int i=0;i<shmOutputs.size();i++)
	    shmOutputs[i]->start();
        }
      }
      catch (...)
      {
        LOG4CPLUS_ERROR(getApplicationLogger(),"Unknown Exception (ShmOutputModule sending InitMsg (pid:"<<getpid() <<")");
      }

      if (forkParams->restart) {
	//do restart things
	scalersUpdates_ = 0;
	try {
	  fsm_.fireEvent("Stop",this); // need to set state in fsm first to allow stopDone transition
	  fsm_.fireEvent("StopDone",this); // need to set state in fsm first to allow stopDone transition
	  fsm_.fireEvent("Enable",this); // need to set state in fsm first to allow stopDone transition
	} catch (...) {
          LOG4CPLUS_WARN(getApplicationLogger(),"Failed to Stop/Enable FSM of the restarted slave EP");
	}
	try{
	  xdata::Serializable *lsid = applicationInfoSpace_->find("lumiSectionIndex");
	  if(lsid) {
	    ((xdata::UnsignedInteger32*)(lsid))->value_--; // need to reset to value before end of ls in which process died
	  }
	}
	catch(...){
	  std::cout << "trouble with lsindex during restart" << std::endl;
	}
	try{
	  xdata::Serializable *lstb = applicationInfoSpace_->find("lsToBeRecovered");
	  if(lstb) {
	    ((xdata::Boolean*)(lstb))->value_ = false; // do not issue eol/bol for all Ls when restarting
	  }
	}
	catch(...){
	  std::cout << "trouble with resetting flag for eol recovery " << std::endl;
	}
	evtProcessor_.adjustLsIndexForRestart();
	evtProcessor_.resetPackedTriggerReport();
      }

      //start other threads
      startReceivingLoop();
      startReceivingMonitorLoop();
      startScalersWorkLoop();
      while(!evtProcessor_.isWaitingForLs())
	::usleep(100000);//wait for scalers loop to start

      //connect DQMShmOutputModule
      if(hasShMem_) attachDqmToShm();

      //catch transition error if we are already Enabled
      try {
        fsm_.fireEvent("EnableDone",this);
      }
      catch (...) {}

      //make sure workloops are started
      while (!wlReceiving_->isActive() || !wlReceivingMonitor_->isActive()) usleep(10000);

      //unmask signals
      sigset_t tmpset_thread;
      sigemptyset(&tmpset_thread);
      sigaddset(&tmpset_thread, SIGQUIT);
      sigaddset(&tmpset_thread, SIGILL);
      sigaddset(&tmpset_thread, SIGABRT);
      sigaddset(&tmpset_thread, SIGFPE);
      sigaddset(&tmpset_thread, SIGSEGV);
      sigaddset(&tmpset_thread, SIGALRM);
      //sigprocmask(SIG_UNBLOCK, &tmpset_thread, 0);
      pthread_sigmask(SIG_UNBLOCK,&tmpset_thread,0);
     
      //set signal handlers 
      struct sigaction sa;
      sigset_t tmpset;
      memset(&tmpset,0,sizeof(tmpset));
      sigemptyset(&tmpset);
      sa.sa_mask=tmpset;
      sa.sa_flags=SA_RESETHAND | SA_SIGINFO;
      sa.sa_handler=0;
      sa.sa_sigaction=evfep_sighandler;

      sigaction(SIGQUIT,&sa,0);
      sigaction(SIGILL,&sa,0);
      sigaction(SIGABRT,&sa,0);
      sigaction(SIGFPE,&sa,0);
      sigaction(SIGSEGV,&sa,0);
      sa.sa_sigaction=evfep_alarmhandler;
      sigaction(SIGALRM,&sa,0);

      //child return to DaqSource
      return ;
    }
    else {

      forkParams->isMaster=1;
      forkInfoObj_->forkParams.slotId=-1;
      if (forkParams->restart) {
	std::ostringstream ost1;
	ost1 << "-I- New Process " << retval << " forked for slot " << forkParams->slotId;
	localLog(ost1.str());
      }
      forkInfoObj_->forkParams.restart=0;
      //start "crash" receiver workloop
    }
  }

  //recreate MessageLogger thread after fork
  try{
    //release the presense factory in master
    edm::PresenceFactory *pf = edm::PresenceFactory::get();
    if(pf != 0) {
      messageServicePresence_ = pf->makePresence("MessageServicePresence");
    }
    else {
      LOG4CPLUS_ERROR(getApplicationLogger(),
	  "Unable to recreate message service presence ");
    }
  }
  catch(...) {
    LOG4CPLUS_ERROR(getApplicationLogger(),"Unknown Exception in MessageServicePresence");
  }
  restart_in_progress_=false;
  edm_init_done_=true;
}

bool FUEventProcessor::enableForkInEDM() 
{
  evtProcessor_.resetWaiting();
  try {
    //set to connect to Shm later
    //if(hasShMem_) setAttachDqmToShm();

    int sc = 0;
    //maybe not needed in MP mode
    evtProcessor_->clearCounters();
    if(isRunNumberSetter_)
      evtProcessor_->setRunNumber(runNumber_.value_);
    else
      evtProcessor_->declareRunNumber(runNumber_.value_);

    //prepare object used to communicate with DaqSource
    pthread_mutex_destroy(&forkObjLock_);
    pthread_mutex_init(&forkObjLock_,0);
    if (forkInfoObj_) delete forkInfoObj_;
    forkInfoObj_ = new moduleweb::ForkInfoObj();
    forkInfoObj_->mst_lock_=&forkObjLock_;
    forkInfoObj_->fuAddr=(void*)this;
    forkInfoObj_->forkHandler = forkProcessFromEDM_helper;
    forkInfoObj_->forkParams.slotId=-1;
    forkInfoObj_->forkParams.restart=0;
    forkInfoObj_->forkParams.isMaster=-1;
    forkInfoObj_->stopCondition=0;
    if (mwrRef_)
      mwrRef_->publishForkInfo(std::string("DaqSource"),forkInfoObj_);

    evtProcessor_->runAsync();
    sc = evtProcessor_->statusAsync();

    if(sc != 0) {
      std::ostringstream oss;
      oss<<"EventProcessor::runAsync returned status code " << sc;
      reasonForFailedState_ = oss.str();
      fsm_.fireFailed(reasonForFailedState_,this);
      LOG4CPLUS_FATAL(log_,reasonForFailedState_);
      return false;
    }
  }
  //catch exceptions on master side
  catch(cms::Exception &e) {
    reasonForFailedState_ = e.explainSelf();
    fsm_.fireFailed(reasonForFailedState_,this);
    LOG4CPLUS_FATAL(log_,reasonForFailedState_);
    return false;
  }    
  catch(std::exception &e) {
    reasonForFailedState_  = e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    LOG4CPLUS_FATAL(log_,reasonForFailedState_);
    return false;
  }
  catch(...) {
    reasonForFailedState_ = "Unknown Exception";
    fsm_.fireFailed(reasonForFailedState_,this);
    LOG4CPLUS_FATAL(log_,reasonForFailedState_);
    return false;
  }
  return true;
}

bool FUEventProcessor::restartForkInEDM(unsigned int slotId) {
  //daqsource will keep this lock until master returns after fork
  //so that we don't do another EP restart in between
  forkInfoObj_->lock();
  forkInfoObj_->forkParams.slotId=slotId;
  forkInfoObj_->forkParams.restart=true;
  forkInfoObj_->forkParams.isMaster=1;
  forkInfoObj_->stopCondition=0;
  LOG4CPLUS_DEBUG(log_, " restarting subprocess in slot "<< slotId <<": posting on control semaphore");
  sem_post(forkInfoObj_->control_sem_);
  forkInfoObj_->unlock();
  usleep(1000);
  //sleep until fork is performed
  int count=50;
  restart_in_progress_=true;
  while (restart_in_progress_ && count--) usleep(20000);
  return true;
}

bool FUEventProcessor::enableClassic()
{
  bool retval = enableCommon();
  while(evtProcessor_->getState()!= edm::event_processor::sRunning){
    LOG4CPLUS_INFO(getApplicationLogger(),"waiting for edm::EventProcessor to start before enabling watchdog");
    ::sleep(1);
  }
  
  //  implementation moved to EPWrapper
  //  startScalersWorkLoop(); // this is now not done any longer 
  localLog("-I- Start completed");
  return retval;
}
bool FUEventProcessor::enableMPEPSlave()
{
  //all this happens only in the child process

  startReceivingLoop();
  startReceivingMonitorLoop();
  evtProcessor_.resetWaiting();
  startScalersWorkLoop();
  while(!evtProcessor_.isWaitingForLs())
    ::sleep(1);

  // @EM test do not run monitor loop in slave, only receiving&Monitor
  //  evtProcessor_.startMonitoringWorkLoop();
  try{
    //    evtProcessor_.makeServicesOnly();
    try{
      edm::PresenceFactory *pf = edm::PresenceFactory::get();
      if(pf != 0) {
	pf->makePresence("MessageServicePresence").release();
      }
      else {
	LOG4CPLUS_ERROR(getApplicationLogger(),
			"Unable to create message service presence ");
      }
    } 
    catch(...) {
      LOG4CPLUS_ERROR(getApplicationLogger(),"Unknown Exception");
    }
  ML::MLlog4cplus::setAppl(this);
  }	  
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "enabling FAILED: " + (std::string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    localLog(reasonForFailedState_);
  }
  bool retval =  enableCommon();
  //  while(evtProcessor_->getState()!= edm::event_processor::sRunning){
  //    LOG4CPLUS_INFO(getApplicationLogger(),"waiting for edm::EventProcessor to start before enabling watchdog");
  //    ::sleep(1);
  //  }
  return retval;
}

bool FUEventProcessor::stopClassic()
{
  bool failed=false;
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start stopping :) ...");
    edm::EventProcessor::StatusCode rc = evtProcessor_.stop();
    if(rc == edm::EventProcessor::epSuccess) 
      fsm_.fireEvent("StopDone",this);
    else
      {
	failed=true;
	//	epMState_ = evtProcessor_->currentStateName();
	if(rc == edm::EventProcessor::epTimedOut)
	  reasonForFailedState_ = "EventProcessor stop timed out";
	else
	  reasonForFailedState_ = "EventProcessor did not receive STOP event";
      }
  }
  catch (xcept::Exception &e) {
    failed=true;
    reasonForFailedState_ = "Stopping FAILED: " + (std::string)e.what();
  }
  catch (edm::Exception &e) {
    failed=true;
    reasonForFailedState_ = "Stopping FAILED: " + (std::string)e.what();
  }
  catch (...) {
    failed=true;
    reasonForFailedState_= "Stopping FAILED: unknown exception";
  }
  try {
    if (hasShMem_) {
      detachDqmFromShm();
      if (failed) 
	LOG4CPLUS_WARN(getApplicationLogger(), 
	    "In failed STOP - success detaching DQM from Shm. pid:" << getpid());
    }
  }
  catch (cms::Exception & e) {
    failed=true;
    reasonForFailedState_= "Stopping FAILED: " + (std::string)e.what();
  }
  catch (...) {
    failed=true;
    reasonForFailedState_= "DQM detach failed: Unknown exception";
  }

  if (failed) {
    LOG4CPLUS_FATAL(getApplicationLogger(),"STOP failed: "
	<< reasonForFailedState_ << " (pid:" << getpid()<<")");
    localLog(reasonForFailedState_);
    fsm_.fireFailed(reasonForFailedState_,this);
  }

  LOG4CPLUS_INFO(getApplicationLogger(),"Finished stopping!");
  localLog("-I- Stop completed");
  return false;
}

void FUEventProcessor::stopSlavesAndAcknowledge()
{
  MsgBuf msg(0,MSQM_MESSAGE_TYPE_STOP);
  MsgBuf msg1(MAX_MSG_SIZE,MSQS_MESSAGE_TYPE_STOP);

  std::vector<bool> processes_to_stop(nbSubProcesses_.value_,false);
  for(unsigned int i = 0; i < subs_.size(); i++)
    {
      pthread_mutex_lock(&stop_lock_);
      if(subs_[i].alive()>0){
	processes_to_stop[i] = true;
	subs_[i].post(msg,false);
      }
      pthread_mutex_unlock(&stop_lock_);
    }
  for(unsigned int i = 0; i < subs_.size(); i++)
    {
      pthread_mutex_lock(&stop_lock_);
      if(processes_to_stop[i]){
	try{
	  subs_[i].rcv(msg1,false);
	}
	catch(evf::Exception &e){
	  std::ostringstream ost;
	  ost << "failed to get STOP - errno ->" << errno << " " << e.what(); 
	  reasonForFailedState_ = ost.str();
	  LOG4CPLUS_ERROR(getApplicationLogger(),reasonForFailedState_);
	  //	  fsm_.fireFailed(reasonForFailedState_,this);
	  localLog(reasonForFailedState_);
	  pthread_mutex_unlock(&stop_lock_);
	  continue;
	}
      }
      else {
	pthread_mutex_unlock(&stop_lock_);
	continue;
      }
      pthread_mutex_unlock(&stop_lock_);
      if(msg1->mtype==MSQS_MESSAGE_TYPE_STOP)
	while(subs_[i].alive()>0) ::usleep(10000);
      subs_[i].disconnect();
    }
  //  subs_.clear();

}

void FUEventProcessor::microState(xgi::Input *in,xgi::Output *out) throw (xgi::exception::Exception)
{
  std::string urn = getApplicationDescriptor()->getURN();
  try{
    evtProcessor_.stateNameFromIndex(0);
    evtProcessor_.moduleNameFromIndex(0);
  if(myProcess_) {std::cout << "microstate called for child! bail out" << std::endl; return;}
  *out << "<tr><td>" << fsm_.stateName()->toString() 
       << "</td><td>"<< (myProcess_ ? "S" : "M") <<"</td><td>" << nblive_ << "</td><td>"
       << nbdead_ << "</td><td><a href=\"/" << urn << "/procStat\">" << getpid() <<"</a></td>";
  evtProcessor_.microState(in,out);
  *out << "<td></td><td>" << nbTotalDQM_ 
       << "</td><td>" << evtProcessor_.getScalersUpdates() << "</td></tr>";
  if(nbSubProcesses_.value_!=0 && !myProcess_) 
    {
      pthread_mutex_lock(&start_lock_);
      for(unsigned int i = 0; i < subs_.size(); i++)
	{
	  try{
	    if(subs_[i].alive()>0)
	      {
		*out << "<tr><td  bgcolor=\"#00FF00\" id=\"a"
		     << i << "\">""Alive</td><td>S</td><td>"
		     << subs_[i].queueId() << "<td>" 
		     << subs_[i].queueStatus()<< "/"
		     << subs_[i].queueOccupancy() << "/"
		     << subs_[i].queuePidOfLastSend() << "/"
		     << subs_[i].queuePidOfLastReceive() 
		     << "</td><td><a id=\"p"<< i << "\" href=\"SubWeb?process=" 
		     << subs_[i].pid() << "&method=procStat\">" 
		     << subs_[i].pid()<<"</a></td>" //<< msg->mtext;
		     << "<td>" << evtProcessor_.stateNameFromIndex(subs_[i].params().Ms) << "</td><td>" 
		     << evtProcessor_.moduleNameFromIndex(subs_[i].params().ms) << "</td><td>" 
		     << subs_[i].params().nba << "/" << subs_[i].params().nbp 
		     << " (" << float(subs_[i].params().nba)/float(subs_[i].params().nbp)*100. <<"%)" 
		     << "</td><td>" << subs_[i].params().ls  << "/" << subs_[i].params().ls 
		     << "</td><td>" << subs_[i].params().ps 
		     << "</td><td" 
		     << ((subs_[i].params().eols<subs_[i].params().ls) ? " bgcolor=\"#00FF00\"" : " bgcolor=\"#FF0000\"")  << ">" 
		     << subs_[i].params().eols  
		     << "</td><td>" << subs_[i].params().dqm 
		     << "</td><td>" << subs_[i].params().trp << "</td>";
	      }
	    else 
	      {
		pthread_mutex_lock(&pickup_lock_);
		*out << "<tr><td id=\"a"<< i << "\" ";
		if(subs_[i].alive()==-1000)
		  *out << " bgcolor=\"#bbaabb\">NotInitialized";
		else
		  *out << (subs_[i].alive()==0 ? ">Done" : " bgcolor=\"#FF0000\">Dead");
		*out << "</td><td>S</td><td>"<< subs_[i].queueId() << "<td>" 
		     << subs_[i].queueStatus() << "/"
		     << subs_[i].queueOccupancy() << "/"
		     << subs_[i].queuePidOfLastSend() << "/"
		     << subs_[i].queuePidOfLastReceive() 
		     << "</td><td id=\"p"<< i << "\">"
		     <<subs_[i].pid()<<"</td><td colspan=\"5\">" << subs_[i].reasonForFailed();
		if(subs_[i].alive()!=0 && subs_[i].alive()!=-1000) 
		  {
		    if(autoRestartSlaves_ && subs_[i].restartCount()<=2) 
		      *out << " will restart in " << subs_[i].countdown() << " s";
		    else if(autoRestartSlaves_)
		      *out << " reached maximum restart count";
		    else *out << " autoRestart is disabled ";
		  }
		*out << "</td><td" 
		     << ((subs_[i].params().eols<subs_[i].params().ls) ? 
			 " bgcolor=\"#00FF00\"" : " bgcolor=\"#FF0000\"")  
		     << ">" 
		     << subs_[i].params().eols  
		     << "</td><td>" << subs_[i].params().dqm 
		     << "</td><td>" << subs_[i].params().trp << "</td>";
		pthread_mutex_unlock(&pickup_lock_);
	      }
	    *out << "</tr>";
	  }
	  catch(evf::Exception &e){
	    *out << "<tr><td id=\"a"<< i << "\" " 
		 <<"bgcolor=\"#FFFF00\">NotResp</td><td>S</td><td>"<< subs_[i].queueId() << "<td>" 
		 << subs_[i].queueStatus() << "/"
		 << subs_[i].queueOccupancy() << "/"
		 << subs_[i].queuePidOfLastSend() << "/"
		 << subs_[i].queuePidOfLastReceive() 
		 << "</td><td id=\"p"<< i << "\">"
		 <<subs_[i].pid()<<"</td>";
	  }
	}
      pthread_mutex_unlock(&start_lock_); 
    }
  }
      catch(evf::Exception &e)
      {
	LOG4CPLUS_INFO(getApplicationLogger(),"evf::Exception caught in microstate - " << e.what());    
      }
    catch(cms::Exception &e)
      {
	LOG4CPLUS_INFO(getApplicationLogger(),"cms::Exception caught in microstate - " << e.what());    
      }
    catch(std::exception &e)
      {
	LOG4CPLUS_INFO(getApplicationLogger(),"std::Exception caught in microstate - " << e.what());    
      }
    catch(...)
      {
	LOG4CPLUS_INFO(getApplicationLogger(),"unknown exception caught in microstate - ");    
      }

}


void FUEventProcessor::updater(xgi::Input *in,xgi::Output *out) throw (xgi::exception::Exception)
{
  using namespace utils;

  *out << updaterStatic_;
  mDiv(out,"loads");
  uptime(out);
  cDiv(out);
  mDiv(out,"st",fsm_.stateName()->toString());
  mDiv(out,"ru",runNumber_.toString());
  mDiv(out,"nsl",nbSubProcesses_.value_);
  mDiv(out,"nsr",nbSubProcessesReporting_.value_);
  mDiv(out,"cl");
  *out << getApplicationDescriptor()->getClassName() 
       << (nbSubProcesses_.value_ > 0 ? "MP " : " ");
  cDiv(out);
  mDiv(out,"in",getApplicationDescriptor()->getInstance());
  if(fsm_.stateName()->toString() != "Halted" && fsm_.stateName()->toString() != "halting"){
    mDiv(out,"hlt");
    *out << "<a href=\"" << configString_.toString() << "\">HLT Config</a>";
    cDiv(out);
    *out << std::endl;
  }
  else
    mDiv(out,"hlt","Not yet...");

  mDiv(out,"sq",squidPresent_.toString());
  mDiv(out,"vwl",(supervising_ ? "Active" : "Not Initialized"));
  mDiv(out,"mwl",evtProcessor_.wlMonitoring());
  if(nbProcessed != 0 && nbAccepted != 0)
    {
      mDiv(out,"tt",((xdata::UnsignedInteger32*)nbProcessed)->value_);
      mDiv(out,"ac",((xdata::UnsignedInteger32*)nbAccepted)->value_);
    }
  else
    {
      mDiv(out,"tt",0);
      mDiv(out,"ac",0);
    }
  if(!myProcess_)
    mDiv(out,"swl",(wlSummarizeActive_ ? "Active" : "Inactive"));
  else
    mDiv(out,"swl",(wlScalersActive_ ? "Active" : "Inactive"));

  mDiv(out,"idi",iDieUrl_.value_);
  if(vp_!=0){
    mDiv(out,"vpi",(unsigned int) vp_);
    if(vulture_->hasStarted()>=0)
      mDiv(out,"vul","Prowling");
    else
      mDiv(out,"vul","Dead");
  }
  else{
    mDiv(out,"vul",(vulture_==0 ? "Nope" : "Hatching"));
  }    
  if(evtProcessor_){
    mDiv(out,"ll");
    *out << evtProcessor_.lastLumi().ls
	 << "," << evtProcessor_.lastLumi().proc << "," << evtProcessor_.lastLumi().acc;
    cDiv(out);
  }
  mDiv(out,"lg");
  for(unsigned int i = logRingIndex_; i<logRingSize_; i++)
    *out << logRing_[i] << std::endl;
  if(logWrap_)
    for(unsigned int i = 0; i<logRingIndex_; i++)
      *out << logRing_[i] << std::endl;
  cDiv(out);
  mDiv(out,"cha");
  if(cpustat_) *out << cpustat_->getChart();
  cDiv(out);
}

void FUEventProcessor::procStat(xgi::Input *in, xgi::Output *out) throw (xgi::exception::Exception)
{
  evf::utils::procStat(out);
}

void FUEventProcessor::sendMessageOverMonitorQueue(MsgBuf &buf)
{
  if(myProcess_) myProcess_->postSlave(buf,true);
}

void FUEventProcessor::makeStaticInfo()
{
  using namespace utils;
  std::ostringstream ost;
  mDiv(&ost,"ve");
  ost<< "$Revision: 1.164 $ (" << edm::getReleaseVersion() <<")";
  cDiv(&ost);
  mDiv(&ost,"ou",outPut_.toString());
  mDiv(&ost,"sh",hasShMem_.toString());
  mDiv(&ost,"mw",hasModuleWebRegistry_.toString());
  mDiv(&ost,"sw",hasServiceWebRegistry_.toString());
  
  xdata::Serializable *monsleep = 0;
  xdata::Serializable *lstimeout = 0;
  try{
    monsleep  = applicationInfoSpace_->find("monSleepSec");
    lstimeout = applicationInfoSpace_->find("lsTimeOut");
  }
  catch(xdata::exception::Exception e){
  }
  
  if(monsleep!=0)
    mDiv(&ost,"ms",monsleep->toString());
  if(lstimeout!=0)
    mDiv(&ost,"lst",lstimeout->toString());
  char cbuf[sizeof(struct utsname)];
  struct utsname* buf = (struct utsname*)cbuf;
  uname(buf);
  mDiv(&ost,"sysinfo");
  ost << buf->sysname << " " << buf->nodename 
      << " " << buf->release << " " << buf->version << " " << buf->machine;
  cDiv(&ost);
  updaterStatic_ = ost.str();
}

void FUEventProcessor::handleSignalSlave(int sig, siginfo_t* info, void* c)
{
  //notify master
  sem_post(sigmon_sem_);

  //sleep while master takes action
  sleep(2);

  //set up alarm if handler deadlocks on unsafe actions
  alarm(5);
  
  std::cout << "--- Slave EP signal handler caught signal " << sig << " process id is " << info->si_pid <<" ---" << std::endl;
  std::cout << "--- Address: " << std::hex << info->si_addr << std::dec << " --- " << std::endl;
  std::cout << "--- Stacktrace follows --" << std::endl;
  std::ostringstream stacktr;
  toolbox::stacktrace(20,stacktr);
  std::cout << stacktr.str();
  if (!rlimit_coresize_changed_)
    std::cout << "--- Dumping core." <<  " --- " << std::endl;
  else
    std::cout << "--- Core dump count exceeded on this FU. ---"<<std::endl;
 
  std::string hasdump = "";
  if (rlimit_coresize_changed_) hasdump = " (core dump disabled) ";

  LOG4CPLUS_ERROR(getApplicationLogger(),    "--- Slave EP signal handler caught signal " << sig << ". process id is " << getpid() 
		                          << " on node " << toolbox::net::getHostName() << " ---" << std::endl
                                          << "--- Address: " << std::hex << info->si_addr << std::dec << " --- " << std::endl
					  << "--- Stacktrace follows"<< hasdump << " ---" << std::endl << stacktr.str()
					  );

  //re-raise signal with default handler (will cause core dump if enabled)
  raise(sig);
}


XDAQ_INSTANTIATOR_IMPL(evf::FUEventProcessor)
