#include "FWEPWrapper.h"

#include "EventFilter/Utilities/interface/ParameterSetRetriever.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "EventFilter/Utilities/interface/ServiceWebRegistry.h"
#include "EventFilter/Utilities/interface/ServiceWeb.h"
#include "EventFilter/Utilities/interface/MicroStateService.h"
#include "EventFilter/Utilities/interface/TimeProfilerService.h"
#include "EventFilter/Modules/interface/ShmOutputModuleRegistry.h"

#include "EventFilter/Modules/src/FUShmOutputModule.h"

#include "toolbox/task/WorkLoopFactory.h"
#include "xdaq/ApplicationDescriptorImpl.h"
#include "xdaq/ContextDescriptor.h"
#include "xdaq/ApplicationContext.h"
#include "xdata/Boolean.h"
#include "xdata/TableIterator.h"
#include "xdata/exdr/Serializer.h"
#include "xdata/exdr/AutoSizeOutputStreamBuffer.h"

#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#undef HAVE_STAT
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/PrescaleService/interface/PrescaleService.h"
#include "FWCore/Framework/interface/TriggerReport.h"

#include "DQMServices/Core/interface/DQMStore.h"


#include "xoap/MessageFactory.h"
#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPBody.h"
#include "xoap/domutils.h"
#include "xoap/Method.h"
#include "xmas/xmas.h"

#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"

#include "utils.icc"

#include <vector>

namespace evf{

  const std::string FWEPWrapper::unknown = "unknown";
  FWEPWrapper::FWEPWrapper(log4cplus::Logger &log, unsigned int instance) 
    : evtProcessor_(0)
    , serviceToken_()
    , servicesDone_(false)
    , epInitialized_(false)
    , prescaleSvc_(0)
    , log_(log)
    , isPython_(true)
    , hasPrescaleService_(false)
    , hasModuleWebRegistry_(false)
    , hasServiceWebRegistry_(false)
    , monitorInfoSpace_(0) 
    , monitorInfoSpaceLegend_(0) 
    , timeoutOnStop_(3)
    , monSleepSec_(1)
    , nbProcessed_(0)
    , nbAccepted_(0)
    , wlMonitoring_(0)
    , asMonitoring_(0)
    , wlMonitoringActive_(false)
    , watching_(false)
    , allPastLumiProcessed_(0)
    , lsid_(0)
    , psid_(0)
    , lsTimeOut_(100000000)
    , lumiSectionIndex_(1)
    , prescaleSetIndex_(0)
    , lastLumiPrescaleIndex_(0)
    , lastLumiUsingEol_(0)
    , lsTimedOut_(false)
    , lsToBeRecovered_(true)
    , scalersUpdateAttempted_(0)
    , scalersUpdateCounter_(0)
    , lumiSectionsCtr_(lsRollSize_+1)
    , lumiSectionsTo_(lsRollSize_)
    , rollingLsIndex_(lsRollSize_-1)
    , rollingLsWrap_(false)
    , rcms_(0)
    , instance_(instance)
    , waitingForLs_(false)
    , mwrRef_(nullptr)
    , sorRef_(nullptr)
    , ftsRef_(nullptr)
    , countDatasets_(false)
  {
    //list of variables for scalers flashlist
    names_.push_back("lumiSectionIndex");
    names_.push_back("prescaleSetIndex");
    names_.push_back("scalersTable");
    namesStatusLegenda_.push_back("macroStateLegenda");
    namesStatusLegenda_.push_back("microStateLegenda");
    namesScalersLegenda_.push_back("scalersLegenda");
    //some initialization of state data
    epMAltState_ = -1;
    epmAltState_ = -1;
    pthread_mutex_init(&ep_guard_lock_,0);
  }

  FWEPWrapper::~FWEPWrapper() {if (0!=evtProcessor_) delete evtProcessor_; evtProcessor_=0;}

  void FWEPWrapper::publishConfigAndMonitorItems(bool multi)
  {

    applicationInfoSpace_->fireItemAvailable("monSleepSec",             &monSleepSec_);
    applicationInfoSpace_->fireItemAvailable("timeoutOnStop",           &timeoutOnStop_);
    applicationInfoSpace_->fireItemAvailable("lsTimeOut",               &lsTimeOut_);

    applicationInfoSpace_->fireItemAvailable("lumiSectionIndex",        &lumiSectionIndex_);
    applicationInfoSpace_->fireItemAvailable("prescaleSetIndex",        &prescaleSetIndex_);
    applicationInfoSpace_->fireItemAvailable("lastLumiPrescaleIndex",   &lastLumiPrescaleIndex_);
    applicationInfoSpace_->fireItemAvailable("lastLumiUsingEol",        &lastLumiUsingEol_);
    applicationInfoSpace_->fireItemAvailable("lsTimedOut",              &lsTimedOut_);
    applicationInfoSpace_->fireItemAvailable("lsToBeRecovered",         &lsToBeRecovered_);

    monitorLegendaInfoSpace_->fireItemAvailable("macroStateLegenda",           &macro_state_legend_);
    monitorLegendaInfoSpace_->fireItemAvailable("microStateLegenda",           &micro_state_legend_);

    monitorInfoSpace_->fireItemAvailable("epMacroState",                &epMState_);
    monitorInfoSpace_->fireItemAvailable("epMicroState",                &epmState_);

    xdata::Table &stbl = trh_.getTable(); 
    scalersInfoSpace_->fireItemAvailable("scalersTable", &stbl);
    scalersInfoSpace_->fireItemAvailable("lumiSectionIndex",      &lumiSectionIndex_);
    scalersInfoSpace_->fireItemAvailable("prescaleSetIndex",      &prescaleSetIndex_);
    scalersInfoSpace_->fireItemAvailable("lastLumiPrescaleIndex", &lastLumiPrescaleIndex_);
    scalersInfoSpace_->fireItemAvailable("lastLumiUsingEol",      &lastLumiUsingEol_);
    scalersLegendaInfoSpace_->fireItemAvailable("scalersLegenda", trh_.getPathLegenda());    

    scalersComplete_.addColumn("instance", "unsigned int 32");
    scalersComplete_.addColumn("lsid", "unsigned int 32");
    scalersComplete_.addColumn("psid", "unsigned int 32");
    scalersComplete_.addColumn("proc", "unsigned int 32");
    scalersComplete_.addColumn("acc",  "unsigned int 32");
    scalersComplete_.addColumn("exprep",  "unsigned int 32");
    scalersComplete_.addColumn("effrep",  "unsigned int 32");
    scalersComplete_.addColumn("triggerReport", "table");  

    xdata::Table::iterator it = scalersComplete_.begin();
    if( it == scalersComplete_.end())
      {
	it = scalersComplete_.append();
	it->setField("instance",instance_);
      }


    //fill initial macrostate legenda information
    unsigned int i = 0;
    std::stringstream oss;
    for(i = (unsigned int)edm::event_processor::sInit; i < (unsigned int)edm::event_processor::sInvalid; i++)
      {
	oss << i << "=" << evtProcessor_->stateName((edm::event_processor::State) i) << " ";
	statmod_.push_back(evtProcessor_->stateName((edm::event_processor::State) i));
      }
    oss << i << "=" << "NotStarted ";
    statmod_.push_back("NotStarted");
    notstarted_state_code_ = i;
    std::stringstream oss2;
    oss2 << 0 << "=Invalid ";
    modmap_["Invalid"]=0;
    mapmod_.resize(1); 
    mapmod_[0]="Invalid";

    monitorInfoSpace_->lock();
    macro_state_legend_ = oss.str();
    micro_state_legend_ = oss2.str();
    monitorInfoSpace_->unlock();

    if(!multi) publishConfigAndMonitorItemsSP();

  }

  void FWEPWrapper::publishConfigAndMonitorItemsSP()
  {
    monitorInfoSpace_->fireItemAvailable("epSPMacroStateInt",             &epMAltState_);
    monitorInfoSpace_->fireItemAvailable("epSPMicroStateInt",             &epmAltState_);

    monitorInfoSpace_->fireItemAvailable("nbProcessed",                 &nbProcessed_);
    monitorInfoSpace_->fireItemAvailable("nbAccepted",                  &nbAccepted_);
  }


  void FWEPWrapper::init(unsigned short serviceMap, std::string &configString)
  {
    hasPrescaleService_ = serviceMap & 0x1;
    hasModuleWebRegistry_ = serviceMap & 0x2;
    hasServiceWebRegistry_ = serviceMap & 0x4;
    bool instanceZero = serviceMap & 0x8;
    hasSubProcesses = serviceMap & 0x10;
    countDatasets_ = (serviceMap&0x20)>0;
    configString_ = configString;
    trh_.resetFormat(); //reset the report table even if HLT didn't change
    scalersUpdateCounter_ = 0;
    if (epInitialized_) {
      LOG4CPLUS_INFO(log_,"CMSSW EventProcessor already initialized: skip!");
      return;
    }
      
    LOG4CPLUS_INFO(log_,"Initialize CMSSW EventProcessor.");
    LOG4CPLUS_INFO(log_,"CMSSW_BASE:"<<getenv("CMSSW_BASE"));
 
    //end job of previous EP instance
    if (0!=evtProcessor_) {
	edm::event_processor::State st = evtProcessor_->getState();
	if(st == edm::event_processor::sJobReady || st == edm::event_processor::sDone) {
	  evtProcessor_->endJob();
	}
        delete evtProcessor_;
	evtProcessor_=0;
    }

    // job configuration string
    ParameterSetRetriever pr(configString_);
    configuration_ = pr.getAsString();
    pathTable_     = pr.getPathTableAsString();
    
    if (configString_.size() > 3 && configString_.substr(configString_.size()-3) == ".py") isPython_ = true;
    boost::shared_ptr<edm::ParameterSet> params; // change this name!
    boost::shared_ptr<std::vector<edm::ParameterSet> > pServiceSets;
    boost::shared_ptr<edm::ProcessDesc> pdesc;
    if(isPython_)
      {
	PythonProcessDesc ppdesc = PythonProcessDesc(configuration_);
	pdesc = ppdesc.processDesc();
      }
    else
      pdesc = boost::shared_ptr<edm::ProcessDesc>(new edm::ProcessDesc(configuration_));
    pServiceSets = pdesc->getServicesPSets();

    // add default set of services
    if(!servicesDone_) {
      //DQMStore should not be created in the Master (MP case) since this poses problems in the slave
      if(!hasSubProcesses){
	internal::addServiceMaybe(*pServiceSets,"DQMStore");
	internal::addServiceMaybe(*pServiceSets,"DQM");
      }
      else{
	internal::removeServiceMaybe(*pServiceSets,"DQMStore");
	internal::removeServiceMaybe(*pServiceSets,"DQM");
      }
      internal::addServiceMaybe(*pServiceSets,"MLlog4cplus");
      internal::addServiceMaybe(*pServiceSets,"MicroStateService");
      internal::addServiceMaybe(*pServiceSets,"ShmOutputModuleRegistry");
      if(hasPrescaleService_) internal::addServiceMaybe(*pServiceSets,"PrescaleService");
      if(hasModuleWebRegistry_) internal::addServiceMaybe(*pServiceSets,"ModuleWebRegistry");
      if(hasServiceWebRegistry_) internal::addServiceMaybe(*pServiceSets,"ServiceWebRegistry");
    
      try{
	serviceToken_ = edm::ServiceRegistry::createSet(*pServiceSets);
	internal::addServiceMaybe(*pServiceSets,"DQMStore");
	internal::addServiceMaybe(*pServiceSets,"DQM");
	//	slaveServiceToken_ = edm::ServiceRegistry::createSet(*pServiceSets);
      }
      catch(cms::Exception &e) {
	LOG4CPLUS_ERROR(log_,e.explainSelf());
      }    
      catch(std::exception &e) {
	LOG4CPLUS_ERROR(log_,e.what());
      }
      catch(...) {
	LOG4CPLUS_ERROR(log_,"Unknown Exception");
      }
      servicesDone_ = true;
    }
  
    edm::ServiceRegistry::Operate operate(serviceToken_);


    //test rerouting of fwk logging to log4cplus
    edm::LogInfo("FWEPWrapper")<<"started MessageLogger Service.";
    edm::LogInfo("FWEPWrapper")<<"Using config \n"<<configuration_;

    DQMStore *dqm = 0;
    try{
      if(edm::Service<DQMStore>().isAvailable())
	dqm = edm::Service<DQMStore>().operator->();
    }
    catch(...) { 
      LOG4CPLUS_INFO(log_,
		     "exception when trying to get service DQMStore");
    }
    if(dqm!=0) dqm->rmdir("");
  

    ModuleWebRegistry *mwr = 0;
    try{
      if(edm::Service<ModuleWebRegistry>().isAvailable())
	mwr = edm::Service<ModuleWebRegistry>().operator->();
    }
    catch(...) { 
      LOG4CPLUS_INFO(log_,
		     "exception when trying to get service ModuleWebRegistry");
    }
    mwrRef_=mwr;

    if(mwr) mwr->clear(); // in case we are coming from stop we need to clear the mwr

    ServiceWebRegistry *swr = 0;
    try{
      if(edm::Service<ServiceWebRegistry>().isAvailable())
	swr = edm::Service<ServiceWebRegistry>().operator->();
    }
    catch(...) { 
      LOG4CPLUS_INFO(log_,
		     "exception when trying to get service ModuleWebRegistry");
    }
    ShmOutputModuleRegistry *sor = 0;
    try{
      if(edm::Service<ShmOutputModuleRegistry>().isAvailable())
	sor = edm::Service<ShmOutputModuleRegistry>().operator->();
    }
    catch(...) { 
      LOG4CPLUS_INFO(log_,
		     "exception when trying to get service ShmOutputModuleRegistry");
    }
    if(sor) sor->clear();
    sorRef_=sor;
    //  if(swr) swr->clear(); // in case we are coming from stop we need to clear the swr


    FastTimerService *fts = 0;
    try{
      if(edm::Service<FastTimerService>().isAvailable())
	fts = edm::Service<FastTimerService>().operator->();
    }
    catch(...) { 
      LOG4CPLUS_INFO(log_,
		     "exception when trying to get service FastTimerService");
    }
    ftsRef_=fts;

    //get and copy streams and datasets PSet from the framework configuration
    edm::ParameterSet streamsPSet;
    edm::ParameterSet datasetsPSet;
    if (countDatasets_)
      try {
        streamsPSet =  pdesc->getProcessPSet()->getParameter<edm::ParameterSet>("streams");
        datasetsPSet =  pdesc->getProcessPSet()->getParameter<edm::ParameterSet>("datasets");
      }
      catch (...) {
        streamsPSet = edm::ParameterSet();
        datasetsPSet = edm::ParameterSet();
      }

    // instantiate the event processor - fatal exceptions are caught in the main application

    std::vector<std::string> defaultServices;
    std::vector<std::string> forcedServices;
    defaultServices.push_back("MessageLogger");
    defaultServices.push_back("InitRootHandlers");
    defaultServices.push_back("JobReportService");
    pdesc->addServices(defaultServices, forcedServices);
    pthread_mutex_lock(&ep_guard_lock_);
    
    evtProcessor_ = new edm::EventProcessor(pdesc,
					    serviceToken_,
					    edm::serviceregistry::kTokenOverrides);
    pthread_mutex_unlock(&ep_guard_lock_);
    //    evtProcessor_->setRunNumber(runNumber_.value_);
    /* removed     
    if(!outPut_)
      evtProcessor_->enableEndPaths(outPut_);    
    outprev_=outPut_;
    */
    // publish all module names to XDAQ infospace
    
    if(mwr) 
      {
	mwr->publish(applicationInfoSpace_);
	mwr->publishToXmas(scalersInfoSpace_);
      }
    if(swr) 
      {
	swr->publish(applicationInfoSpace_);
      }
    if (sor && countDatasets_)
      {
        sor->insertStreamAndDatasetInfo(streamsPSet,datasetsPSet);
	sor->updateDatasetInfo();
      }
    // get the prescale service
    LOG4CPLUS_INFO(log_,
		   "Checking for edm::service::PrescaleService!");
    try {
      if(edm::Service<edm::service::PrescaleService>().isAvailable())
	{
	  LOG4CPLUS_INFO(log_,
			 "edm::service::PrescaleService is available!");
	  prescaleSvc_ = edm::Service<edm::service::PrescaleService>().operator->();
	  LOG4CPLUS_INFO(log_,
			 "Obtained pointer to PrescaleService");
	}
    }
    catch(...) {
      LOG4CPLUS_INFO(log_,
		     "exception when trying to get service "
		     <<"edm::service::PrescaleService");
    }
    const edm::ParameterSet *prescaleSvcConfig = internal::findService(*pServiceSets,"PrescaleService");
    if(prescaleSvc_ != 0 && prescaleSvcConfig !=0) prescaleSvc_->reconfigure(*prescaleSvcConfig);
  
    monitorLegendaInfoSpace_->lock();
    //fill microstate legenda information
    descs_ = evtProcessor_->getAllModuleDescriptions();

    std::stringstream oss2;
    unsigned int outcount = 0;
    oss2 << 0 << "=Invalid ";
    oss2 << 1 << "=FwkOvh ";
    oss2 << 2 << "=Input ";
    modmap_["Invalid"]=0;
    modmap_["FWKOVH"]=1;
    modmap_["INPUT"]=2;
    mapmod_.resize(descs_.size()+4); // all modules including output plus one input plus DQM plus the invalid state 0
    mapmod_[0]="Invalid";
    mapmod_[1]="FWKOVH";
    mapmod_[2]="INPUT";
    outcount+=2;
    for(unsigned int j = 0; j < descs_.size(); j++)
      {
	if(descs_[j]->moduleName() == "ShmStreamConsumer") // find something better than hardcoding name
	  { 
	    outcount++;
	    oss2 << outcount << "=" << descs_[j]->moduleLabel() << " ";
	    modmap_[descs_[j]->moduleLabel()]=outcount;
	    mapmod_[outcount] = descs_[j]->moduleLabel();
	  }
      }
    modmap_["DQM"]=outcount+1;
    mapmod_[outcount+1]="DQM";
    oss2 << outcount+1 << "=DQMHistograms ";
    unsigned int modcount = 1;
    for(unsigned int i = 0; i < descs_.size(); i++)
      {
	if(descs_[i]->moduleName() != "ShmStreamConsumer")
	  {
	    modcount++;
	    oss2 << outcount+modcount << "=" << descs_[i]->moduleLabel() << " ";
	    modmap_[descs_[i]->moduleLabel()]=outcount+modcount;
	    mapmod_[outcount+modcount] = descs_[i]->moduleLabel();
	  }
      }
//     std::cout << "*******************************microstate legend**************************" << std::endl;
//     std::cout << oss2.str() << std::endl;
//     std::cout << "*******************************microstate legend**************************" << std::endl;

    if(instanceZero){
      micro_state_legend_ = oss2.str().c_str();
    }
    monitorLegendaInfoSpace_->unlock();
    try{
      monitorLegendaInfoSpace_->fireItemGroupChanged(namesStatusLegenda_,0);
      scalersLegendaInfoSpace_->fireItemGroupChanged(namesScalersLegenda_,0);
      ::usleep(10);
    }
    catch(xdata::exception::Exception &e)
      {
	LOG4CPLUS_ERROR(log_, "Exception from fireItemGroupChanged: " << e.what());
      }
    LOG4CPLUS_INFO(log_," edm::EventProcessor configuration finished.");
    edm::TriggerReport tr;
    evtProcessor_->getTriggerReport(tr);
    trh_.formatReportTable(tr,descs_,pathTable_,instanceZero);
    epInitialized_ = true;
    return;
  }

  //______________________________________________________________________________
  void FWEPWrapper::makeServicesOnly()
  {
    edm::ServiceRegistry::Operate operate(serviceToken_);
  }
 
  //______________________________________________________________________________
  ModuleWebRegistry * FWEPWrapper::getModuleWebRegistry()
  {
    return mwrRef_;
  }

  //______________________________________________________________________________
  ShmOutputModuleRegistry * FWEPWrapper::getShmOutputModuleRegistry()
  {
    return sorRef_;
  }

  //______________________________________________________________________________
  void FWEPWrapper::setupFastTimerService(unsigned int nProcesses)
  {
    if (ftsRef_) ftsRef_->setNumberOfProcesses( nProcesses );
  }
 
  //______________________________________________________________________________
  edm::EventProcessor::StatusCode FWEPWrapper::stop()
  {
    edm::event_processor::State st = evtProcessor_->getState();
    
    LOG4CPLUS_WARN(log_,"FUEventProcessor::stopEventProcessor.1 state "
		   << evtProcessor_->stateName(st));
    edm::EventProcessor::StatusCode rc = edm::EventProcessor::epSuccess;

    //total stopping time allowed before epTimeout/epOther
    unsigned int stopTimeLeft = (timeoutOnStop_.value_+1)*1000000;
    if (timeoutOnStop_.value_==0) stopTimeLeft=1500000;
 
    while (!(st==edm::event_processor::sStopping || st==edm::event_processor::sJobReady
			             || st==edm::event_processor::sDone)) {
      usleep(100000);
      st = evtProcessor_->getState();
      if (stopTimeLeft<500000) {
        break;
      }
      stopTimeLeft-=100000;
    }
    //if already in stopped state
    if (st==edm::event_processor::sJobReady || st==edm::event_processor::sDone) 
      return edm::EventProcessor::epSuccess;

    //if not even in stopping state
    if(st!=edm::event_processor::sStopping) {
      LOG4CPLUS_WARN(log_,
	  "FUEventProcessor::stopEventProcessor.2 After 1s - state: "
	  << evtProcessor_->stateName(st)); 
      return edm::EventProcessor::epOther;
    }
    LOG4CPLUS_WARN(log_,"FUEventProcessor::stopEventProcessor.3 state "
	<< evtProcessor_->stateName(st));

    //use remaining time left for the framework timeout
    if (stopTimeLeft<1000000) stopTimeLeft=1000000;
    stopTimeLeft/=1000000;
    if (timeoutOnStop_.value_==0) stopTimeLeft=0;

    try  {
      rc = evtProcessor_->waitTillDoneAsync(stopTimeLeft);
      watching_ = false;
    }
    catch(cms::Exception &e) {
      XCEPT_RAISE(evf::Exception,e.explainSelf());
    }    
    catch(std::exception &e) {
      XCEPT_RAISE(evf::Exception,e.what());
    }
    catch(...) {
      XCEPT_RAISE(evf::Exception,"Unknown Exception");
    }
    allPastLumiProcessed_ = 0;
    return rc;
    
  }

  void FWEPWrapper::stopAndHalt()
  {
    edm::ServiceRegistry::Operate operate(serviceToken_);
    ModuleWebRegistry *mwr = 0;
    try{
      if(edm::Service<ModuleWebRegistry>().isAvailable())
	mwr = edm::Service<ModuleWebRegistry>().operator->();
    }
    catch(...) { 
      LOG4CPLUS_INFO(log_,
		     "exception when trying to get service ModuleWebRegistry");
    }
    
    if(mwr) 
      {
	mwr->clear();
      }
    
    ServiceWebRegistry *swr = 0;
    try{
      if(edm::Service<ServiceWebRegistry>().isAvailable())
	swr = edm::Service<ServiceWebRegistry>().operator->();
    }
    catch(...) { 
      LOG4CPLUS_INFO(log_,
		     "exception when trying to get service ModuleWebRegistry");
    }
    
    if(swr) 
      {
	swr->clear();
      }

    edm::event_processor::State st = evtProcessor_->getState();
    edm::EventProcessor::StatusCode rc = stop();
    watching_ = false;
    if(rc != edm::EventProcessor::epTimedOut)
      {
	if(st == edm::event_processor::sJobReady || st == edm::event_processor::sDone)
	  evtProcessor_->endJob();
	pthread_mutex_lock(&ep_guard_lock_);
	delete evtProcessor_;
	evtProcessor_ = 0;
	pthread_mutex_unlock(&ep_guard_lock_);
	epInitialized_ = false;
      }
    else
      {
	XCEPT_RAISE(evf::Exception,"EventProcessor stop timed out");
      }
    allPastLumiProcessed_ = 0;    
  }
    
  void FWEPWrapper::startMonitoringWorkLoop() throw (evf::Exception)
  {
    pid_t pid = getpid();
    nbProcessed_.value_ = 0;
    nbAccepted_.value_ = 0;
    struct timezone timezone;
    gettimeofday(&monStartTime_,&timezone);

    std::ostringstream ost;
    ost << "Monitoring" << pid;
    try {
      wlMonitoring_=
	toolbox::task::getWorkLoopFactory()->getWorkLoop(ost.str().c_str(),
							 "waiting");

      if (!wlMonitoring_->isActive()) wlMonitoring_->activate();
      asMonitoring_ = toolbox::task::bind(this,&FWEPWrapper::monitoring,
					  ost.str().c_str());

      wlMonitoring_->submit(asMonitoring_);
      wlMonitoringActive_ = true;

    }
    catch (xcept::Exception& e) {
      std::string msg = "Failed to start workloop 'Monitoring'.";

      XCEPT_RETHROW(evf::Exception,msg,e);
    }
  }



  //______________________________________________________________________________
  bool FWEPWrapper::monitoring(toolbox::task::WorkLoop* wl)
  {
  
    struct timeval  monEndTime;
    struct timezone timezone;
    gettimeofday(&monEndTime,&timezone);
    edm::ServiceRegistry::Operate operate(serviceToken_);
    MicroStateService *mss = 0;
    if(!hasSubProcesses) monitorInfoSpace_->lock();
    if(evtProcessor_)
      {
	epMState_ = evtProcessor_->currentStateName();
	epMAltState_ = (int) evtProcessor_->getState();
      }
    else
      {
	epMState_ = "Off";
	epMAltState_ = -1;
      }
    if(0 != evtProcessor_ && evtProcessor_->getState() != edm::event_processor::sInit)
      {
	try{
	  mss = edm::Service<MicroStateService>().operator->();
	}
	catch(...) { 
	  LOG4CPLUS_INFO(log_,
			 "exception when trying to get service MicroStateService");
	}
	lsid_ = lumiSectionIndex_.value_;
	psid_ = prescaleSetIndex_.value_;
      }
    if(mss) 
      {
	epmState_  = mss->getMicroState2();
	epmAltState_ = modmap_[mss->getMicroState2()];
      }
    if(evtProcessor_)
      {
	nbProcessed_ = evtProcessor_->totalEvents();
	nbAccepted_  = evtProcessor_->totalEventsPassed(); 
      }
    if(!hasSubProcesses) monitorInfoSpace_->unlock();  

    ::sleep(monSleepSec_.value_);
    return true;
  }

  bool FWEPWrapper::getTriggerReport(bool useLock)
  {
    edm::ServiceRegistry::Operate operate(serviceToken_);
    // Calling this method results in calling 
    // evtProcessor_->getTriggerReport, the value returned is encoded as
    // a xdata::Table.

    LOG4CPLUS_DEBUG(log_,"getTriggerReport action invoked");

    //Get the trigger report.
    ModuleWebRegistry *mwr = 0;
    try{
      if(edm::Service<ModuleWebRegistry>().isAvailable())
	mwr = edm::Service<ModuleWebRegistry>().operator->();
    }
    catch(...) { 
      LOG4CPLUS_INFO(log_,
		     "exception when trying to get service ModuleWebRegistry");
      return false;
    }
    edm::TriggerReport tr; 
    if(mwr==0) return false;

    unsigned int ls = 0;
    unsigned int ps = 0;
    timeval tv;
    if(useLock) {
      gettimeofday(&tv,0);
      //      std::cout << getpid() << " calling openBackdoor " << std::endl;
      //waitingForLs_ = true;//moving this behind mutex lock
      mwr->openBackDoor("DaqSource",lsTimeOut_,&waitingForLs_);
      //      std::cout << getpid() << " opened Backdoor " << std::endl;
    }

    xdata::Table::iterator it = scalersComplete_.begin();
    ps = lastLumiPrescaleIndex_.value_;
    //	if(prescaleSvc_ != 0) prescaleSvc_->setIndex(ps);
    it->setField("psid",lastLumiPrescaleIndex_);
    psid_ = prescaleSetIndex_.value_;
    if(prescaleSvc_ != 0) prescaleSvc_->setIndex(psid_);
    ls = lumiSectionIndex_.value_;
    localLsIncludingTimeOuts_.value_ = ls;
    it->setField("lsid", localLsIncludingTimeOuts_);

    lsTriplet lst;
    lst.ls = localLsIncludingTimeOuts_.value_;
    lst.proc = evtProcessor_->totalEvents()-allPastLumiProcessed_;
    lst.acc = evtProcessor_->totalEventsPassed()-
      (rollingLsWrap_ ? lumiSectionsCtr_[0].acc : lumiSectionsCtr_[rollingLsIndex_+1].acc);
       lumiSectionsCtr_[rollingLsIndex_] = lst;
    allPastLumiProcessed_ = evtProcessor_->totalEvents();


    evtProcessor_->getTriggerReport(tr);

    if(useLock){
      //      std::cout << getpid() << " calling closeBackdoor " << std::endl;
      mwr->closeBackDoor("DaqSource");
      //      std::cout << getpid() << " closed Backdoor " << std::endl;
    }  

    trh_.formatReportTable(tr,descs_,pathTable_,false);


    trh_.triggerReportUpdate(tr,ls,ps,trh_.checkLumiSection(ls));
    ShmOutputModuleRegistry *sor = 0;
    try{
      if(edm::Service<ShmOutputModuleRegistry>().isAvailable())
	sor = edm::Service<ShmOutputModuleRegistry>().operator->();
    }
    catch(...) { 
      LOG4CPLUS_INFO(log_,
		     "exception when trying to get service ShmOutputModuleRegistry");
      return false;
    }


    trh_.packTriggerReport(tr,sor,countDatasets_);
    it->setField("triggerReport",trh_.getTableWithNames());
    //    std::cout << getpid() << " returning normally from gettriggerreport " << std::endl;
    return true;
  }

  bool FWEPWrapper::fireScalersUpdate()
  {
    //    trh_.printReportTable();
    scalersUpdateAttempted_++;
    //@@EM commented out on
    // @@EM 21.06.2011 - this flashlist is too big to be handled by LAS 
    /*
    try{
      //      scalersInfoSpace_->unlock();
      // scalersInfoSpace_->fireItemGroupChanged(names_,0); 
      ::usleep(10);
      //      scalersInfoSpace_->lock();
    }
    catch(xdata::exception::Exception &e)
      {
	LOG4CPLUS_ERROR(log_, "Exception from fireItemGroupChanged: " << e.what());
	//	localLog(e.what());
	return false;
      }
    */
    //@@EM added on 21.06.2011 
    // refresh the microstate legenda every 10 lumisections
    if(scalersUpdateAttempted_%10 == 0)
      monitorLegendaInfoSpace_->fireItemGroupChanged(namesStatusLegenda_,0);
    
    //if there is no state listener then do not attempt to send to monitorreceiver
    if(rcms_==0) return false;
    try{
      if(trh_.getProcThisLumi()!=0U)
	createAndSendScalersMessage();
      scalersUpdateCounter_++;
    }
    catch(...){return false;}
    return true;
  }


  //______________________________________________________________________________
  void FWEPWrapper::summaryWebPage(xgi::Input *in, xgi::Output *out,const std::string &urn)
  {
    //    std::string urn = xappDesc_->getURN();

    *out << "<table>"                                                  << std::endl;
    
    *out << "<tr valign=\"top\">"                                      << std::endl;
    *out << "<td>" << std::endl;
    
    TriggerReportStatic *tr = (TriggerReportStatic *)(trh_.getPackedTriggerReport()->mtext);
    // trigger summary table
    *out << "<table border=1 bgcolor=\"#CFCFCF\">" << std::endl;
    *out << "  <tr>"							<< std::endl;
    *out << "    <th colspan=7>"						<< std::endl;
    *out << "      " << "Trigger Summary up to LS "
	 << trh_.getLumiSectionReferenceIndex()-1 << std::endl;
    *out << "    </th>"							<< std::endl;
    *out << "  </tr>"							<< std::endl;
    
    *out << "  <tr >"							<< std::endl;
    *out << "    <th >Path</th>"						<< std::endl;
    *out << "    <th >Exec</th>"						<< std::endl;
    *out << "    <th >Pass</th>"						<< std::endl;
    *out << "    <th >Fail</th>"						<< std::endl;
    *out << "    <th >Except</th>"					<< std::endl;
    *out << "  </tr>"							<< std::endl;
    
    
    for(int i=0; i<tr->trigPathsInMenu; i++) {
      *out << "  <tr>" << std::endl;
      *out << "    <td>"<< i << "</td>"		<< std::endl;
      *out << "    <td>" << trh_.getl1pre(i) << "</td>"		<< std::endl;
      *out << "    <td>" << trh_.getaccept(i) << "</td>"	<< std::endl;
      *out << "    <td >" << trh_.getfailed(i) << "</td>"	<< std::endl;
      *out << "    <td ";
      if(trh_.getexcept(i) !=0)
	*out << "bgcolor=\"red\""		      					<< std::endl;
      *out << ">" << trh_.getexcept(i) << "</td>"		<< std::endl;
      *out << "  </tr >"								<< std::endl;
      
      }
    *out << "  <tr><th colspan=7>EndPaths</th></tr>"		<< std::endl;

    for(int i=tr->trigPathsInMenu; i<tr->endPathsInMenu + tr->trigPathsInMenu; i++) {
      *out << "  <tr>" << std::endl;
      *out << "    <td>"<< i << "</td>"		<< std::endl;
      *out << "    <td>" << trh_.getl1pre(i) << "</td>"		<< std::endl;
      *out << "    <td>" << trh_.getaccept(i) << "</td>"	<< std::endl;
      *out << "    <td >" << trh_.getfailed(i) << "</td>"	<< std::endl;
      *out << "    <td ";
      if(trh_.getexcept(i) !=0)
	*out << "bgcolor=\"red\""		      					<< std::endl;
      *out << ">" << trh_.getexcept(i) << "</td>"		<< std::endl;
      *out << "  </tr >"								<< std::endl;
      
      }


    *out << "</table>" << std::endl;
    *out << "</td>" << std::endl;
    *out << "</tr>" << std::endl;
    *out << "</table>" << std::endl;
  }


  //______________________________________________________________________________
  void FWEPWrapper::taskWebPage(xgi::Input *in, xgi::Output *out,const std::string &urn)
  {
    //    std::string urn = xappDesc_->getURN();
    ModuleWebRegistry *mwr = 0;
    edm::ServiceRegistry::Operate operate(evtProcessor_->getToken());
    try{
      if(edm::Service<ModuleWebRegistry>().isAvailable())
	mwr = edm::Service<ModuleWebRegistry>().operator->();
    }
    catch(...) {
      LOG4CPLUS_WARN(log_,
		     "Exception when trying to get service ModuleWebRegistry");
    }
    TimeProfilerService *tpr = 0;
    try{
      if(edm::Service<TimeProfilerService>().isAvailable())
	tpr = edm::Service<TimeProfilerService>().operator->();
    }
    catch(...) { 
    }

    *out << "<table>"                                                  << std::endl;
    
    *out << "<tr valign=\"top\">"                                      << std::endl;
    *out << "<td>" << std::endl;
    
    
    edm::TriggerReport tr; 
    evtProcessor_->getTriggerReport(tr);
    
    // trigger summary table
    *out << "<table border=1 bgcolor=\"#CFCFCF\">" << std::endl;
    *out << "  <tr>"							<< std::endl;
    *out << "    <th colspan=7>"						<< std::endl;
    *out << "      " << "Trigger Summary"					<< std::endl;
    *out << "    </th>"							<< std::endl;
    *out << "  </tr>"							<< std::endl;
	
    *out << "  <tr >"							<< std::endl;
    *out << "    <th >Path</th>"						<< std::endl;
    *out << "    <th >Exec</th>"						<< std::endl;
    *out << "    <th >Pass</th>"						<< std::endl;
    *out << "    <th >Fail</th>"						<< std::endl;
    *out << "    <th >Except</th>"					<< std::endl;
    *out << "    <th >TargetPF</th>"					<< std::endl;
    *out << "  </tr>"							<< std::endl;
    xdata::Serializable *psid = 0;
    try{
      psid = applicationInfoSpace_->find("prescaleSetIndex");
    }
    catch(xdata::exception::Exception e){
    }
    ShmOutputModuleRegistry *sor = 0;
    try{
      if(edm::Service<ShmOutputModuleRegistry>().isAvailable())
	sor = edm::Service<ShmOutputModuleRegistry>().operator->();
    }
    catch(...) { 
      LOG4CPLUS_INFO(log_,
		     "exception when trying to get service ShmOutputModuleRegistry");
    }

    
    for(unsigned int i=0; i<tr.trigPathSummaries.size(); i++) {
      *out << "  <tr>" << std::endl;
      *out << "    <td>"<< tr.trigPathSummaries[i].name << "</td>"		<< std::endl;
      *out << "    <td>" << tr.trigPathSummaries[i].timesRun << "</td>"		<< std::endl;

      *out << "    <td>" << tr.trigPathSummaries[i].timesPassed << "</td>"	<< std::endl;
      *out << "    <td >" << tr.trigPathSummaries[i].timesFailed << "</td>"	<< std::endl;

      *out << "    <td ";
      if(tr.trigPathSummaries[i].timesExcept !=0)
	*out << "bgcolor=\"red\""		      					<< std::endl;
      *out << ">" << tr.trigPathSummaries[i].timesExcept << "</td>"		<< std::endl;
      if(psid != 0)
	{
	  *out << "    <td>"
	       << prescaleSvc_->getPrescale(tr.trigPathSummaries[i].name) 
	       << "</td>"		<< std::endl;
	}
      else 	*out << "    <td>N/A</td>"		                        << std::endl;
      *out << "  </tr >"								<< std::endl;
      
    }



    for(unsigned int i=0; i<tr.endPathSummaries.size(); i++) {
      std::string olab = trh_.findLabelOfModuleTypeInEndPath(tr,descs_,
							     i,"ShmStreamConsumer");
      evf::OutputModule *o = sor->get(olab);
      *out << "  <tr>" << std::endl;
      *out << "    <td>"<< tr.endPathSummaries[i].name << "</td>"		<< std::endl;
      *out << "    <td>" << tr.endPathSummaries[i].timesRun << "</td>"		<< std::endl;
      *out << "    <td>" << (o ? o->getCounts() : -1) << "</td>"	<< std::endl;
      *out << "    <td >" << (o ? (tr.endPathSummaries[i].timesRun - o->getCounts()) : -1) << "</td>"	<< std::endl;
      *out << "    <td ";
      if(tr.endPathSummaries[i].timesExcept !=0)
	*out << "bgcolor=\"red\""		      					<< std::endl;
      *out << ">" << tr.endPathSummaries[i].timesExcept << "</td>"		<< std::endl;
      *out << "    <td>N/A</td>"		                        << std::endl;
      *out << "  </tr >"								<< std::endl;
      
    }
  
    *out << "</table>" << std::endl;
    
    *out << "</td>" << std::endl;
    

    
    *out << "<td>" << std::endl;
    //Process details table
    *out << "<table frame=\"void\" rules=\"rows\" class=\"modules\">"	<< std::endl;
    *out << "  <tr>"							<< std::endl;
    *out << "    <th colspan=3>"						<< std::endl;
    *out << "      " << "HLT"						<< std::endl;
    if(descs_.size()>0)
      *out << " (Process " << descs_[0]->processName() << ")"		<< std::endl;
    *out << "    </th>"							<< std::endl;
    *out << "  </tr>"							<< std::endl;

    *out << "  <tr >"							<< std::endl;
    *out << "    <th >"							<< std::endl;
    *out << "       Module"						<< std::endl;
    *out << "    </th>"							<< std::endl;
    *out << "    <th >"							<< std::endl;
    *out << "       Label"						<< std::endl;
    *out << "    </th>"							<< std::endl;
    *out << "    <th >"							<< std::endl;
    *out << "       Version"						<< std::endl;
    *out << "    </th>"							<< std::endl;
    if(tpr)
      {
	*out << "    <th >"                                                       << std::endl;
	*out << "       first"                                            << std::endl;
	*out << "    </th>"                                                       << std::endl;
	*out << "    <th >"                                                       << std::endl;
	*out << "       ave"                                              << std::endl;
	*out << "    </th>"                                                       << std::endl;
	*out << "    <th >"                                                       << std::endl;
	*out << "       max"                                              << std::endl;
	*out << "    </th>"                                                       << std::endl;
      }
    *out << "  </tr>"							<< std::endl;
    if(mwr && mwr->checkWeb("DaqSource"))
      *out << "    <tr><td ><a href=\"/" << urn 
	   << "module=DaqSource\">DaqSource</a> </td></tr>";
    
    for(unsigned int idesc = 0; idesc < descs_.size(); idesc++)
      {
	*out << "  <tr>"							<< std::endl;
	*out << "    <td >";
	if(mwr && mwr->checkWeb(descs_[idesc]->moduleName()))
	  *out << "<a href=\"/" << urn 
	       << "module=" 
	       << descs_[idesc]->moduleName() << "\">" 
	       << descs_[idesc]->moduleName() << "</a>";
	else
	  *out << descs_[idesc]->moduleName();
	*out << "</td>"							<< std::endl;
	*out << "    <td >";
	*out << descs_[idesc]->moduleLabel();
	*out << "</td>"							<< std::endl;
	*out << "    <td >";
	*out << descs_[idesc]->releaseVersion();
	*out << "</td>"							<< std::endl;
	if(tpr)
	  {
	    *out << "    <td align=\"right\">";
	    *out << tpr->getFirst(descs_[idesc]->moduleLabel());
	    *out << "</td>"                                                       << std::endl;
	    *out << "    <td align=\"right\"";
	    *out << (tpr->getAve(descs_[idesc]->moduleLabel())>1. ? "bgcolor=\"red\"" : "") 
		 << ">";
	    *out << tpr->getAve(descs_[idesc]->moduleLabel());
	    *out << "</td>"                                                       << std::endl;
	    *out << "    <td align=\"right\">";
	    *out << tpr->getMax(descs_[idesc]->moduleLabel());
	    *out << "</td>"                                                       << std::endl;
	  }
	*out << "  </tr>" << std::endl;
      }
    *out << "</table>" << std::endl;
    *out << "</td>" << std::endl;
  }

  //______________________________________________________________________________
  void FWEPWrapper::moduleWeb(xgi::Input  *in, xgi::Output *out)
  {
    using namespace cgicc;
    Cgicc cgi(in);
    std::vector<FormEntry> el1;
    cgi.getElement("module",el1);
    if(evtProcessor_)  {
      if(el1.size()!=0) {
	std::string mod = el1[0].getValue();
	edm::ServiceRegistry::Operate operate(evtProcessor_->getToken());
	ModuleWebRegistry *mwr = 0;
	try{
	  if(edm::Service<ModuleWebRegistry>().isAvailable())
	    mwr = edm::Service<ModuleWebRegistry>().operator->();
	}
	catch(...) { 
	  LOG4CPLUS_WARN(log_,
			 "Exception when trying to get service ModuleWebRegistry");
	}
	mwr->invoke(in,out,mod);
      }
    }
    else {
      *out<<"EventProcessor just disappeared "<<std::endl;
    }
  }
  

  //______________________________________________________________________________
  void FWEPWrapper::serviceWeb(xgi::Input  *in, xgi::Output *out)
  {
    using namespace cgicc;
    Cgicc cgi(in);
    std::vector<FormEntry> el1;
    cgi.getElement("service",el1);
    if(evtProcessor_)  {
      if(el1.size()!=0) {
	std::string ser = el1[0].getValue();
	edm::ServiceRegistry::Operate operate(evtProcessor_->getToken());
	ServiceWebRegistry *swr = 0;
	try{
	  if(edm::Service<ServiceWebRegistry>().isAvailable())
	    swr = edm::Service<ServiceWebRegistry>().operator->();
	}
	catch(...) { 
	  LOG4CPLUS_WARN(log_,
			 "Exception when trying to get service ModuleWebRegistry");
	}
	swr->invoke(in,out,ser);
      }
    }
    else {
      *out<<"EventProcessor just disappeared "<<std::endl;
    }
  }

//______________________________________________________________________________
  void FWEPWrapper::microState(xgi::Input  *in, xgi::Output *out)
  {
    edm::ServiceRegistry::Operate operate(serviceToken_);
    MicroStateService *mss = 0;
    std::string micro1 = "unavailable";
    if(epInitialized_)
      micro1 = "initialized";
    std::string micro2 = "unavailable";
    if(evtProcessor_!=0 && evtProcessor_->getState() != edm::event_processor::sInit)
      {
	try{
	mss = edm::Service<MicroStateService>().operator->();
	}
	catch(...) { 
	  LOG4CPLUS_INFO(log_,
			 "exception when trying to get service MicroStateService");
	}
	pthread_mutex_lock(&ep_guard_lock_);
	if(evtProcessor_!=0) micro1 = evtProcessor_->currentStateName();
	pthread_mutex_unlock(&ep_guard_lock_);
      }

    if(mss) {
      micro2 = mss->getMicroState2();
    }

    //    *out << fsm_.stateName()->toString() << std::endl;   
    *out << "<td>" << micro1 << "</td>";
    *out << "<td>" << micro2 << "</td>";
    *out << "<td>" << nbAccepted_.value_ << "/" << nbProcessed_.value_  
	 << " (" << float(nbAccepted_.value_)/float(nbProcessed_.value_)*100. <<"%)" << "</td>";
    *out << "<td>" << lsid_ << "/" << lsidTimedOutAsString_ << "</td>";
    *out << "<td>" << psid_ << "</td>";
    

  }
  
  void FWEPWrapper::lumiSumTable(xgi::Output *out)
  {
    // lumisection summary table
    *out << "   <table border=1 bgcolor=\"#CFCFCF\">" << std::endl;
    *out << "     <tr>"							<< std::endl;
    *out << "       <td> LS </td>";
    if(rollingLsWrap_)
      {
	for(unsigned int i = rollingLsIndex_; i < lumiSectionsCtr_.size(); i++)
	  *out << "<td " << (lumiSectionsTo_[i] ? "bgcolor=\"red\"" : "")
	       << ">" << lumiSectionsCtr_[i].ls << "</td>" << std::endl;
	for(unsigned int i = 0; i < rollingLsIndex_; i++)
	  *out << "<td " << (lumiSectionsTo_[i] ? "bgcolor=\"red\"" : "")
	       << ">" << lumiSectionsCtr_[i].ls << "</td>" << std::endl;
      }
    else
      for(unsigned int i = rollingLsIndex_; i < lumiSectionsCtr_.size(); i++)
	*out << "<td  " << (lumiSectionsTo_[i] ? "bgcolor=\"red\"" : "")
	     << ">" << lumiSectionsCtr_[i].ls << "</td>" << std::endl;
    
    *out << "     </tr>"							<< std::endl;    
    *out << "     <tr>"							<< std::endl;
    *out << "       <td> Ev </td>";
    if(rollingLsWrap_)
      {
	for(unsigned int i = rollingLsIndex_; i < lumiSectionsCtr_.size(); i++)
	  *out << "<td>" << lumiSectionsCtr_[i].proc << "</td>" << std::endl;
	for(unsigned int i = 0; i < rollingLsIndex_; i++)
	  *out << "<td>" << lumiSectionsCtr_[i].proc << "</td>" << std::endl;
      }
    else
      for(unsigned int i = rollingLsIndex_; i < lumiSectionsCtr_.size(); i++)
	*out << "<td>" << lumiSectionsCtr_[i].proc << "</td>" << std::endl;
    *out << "     </tr>"							<< std::endl;    
    *out << "     <tr>"							<< std::endl;
    *out << "       <td> Acc </td>";
    if(rollingLsWrap_)
      {
	for(unsigned int i = rollingLsIndex_; i < lumiSectionsCtr_.size(); i++)
	  *out << "<td>" << lumiSectionsCtr_[i].acc << "</td>" << std::endl;
	for(unsigned int i = 0; i < rollingLsIndex_; i++)
	  *out << "<td>" << lumiSectionsCtr_[i].acc << "</td>" << std::endl;
      }
    else
      for(unsigned int i = rollingLsIndex_; i < lumiSectionsCtr_.size(); i++)
	*out << "<td>" << lumiSectionsCtr_[i].acc << "</td>" << std::endl;
    *out << "     </tr>"							<< std::endl;    
    *out << "</table>" << std::endl;
  }


  void FWEPWrapper::sumAndPackTriggerReport(MsgBuf &buf)
  {
    trh_.sumAndPackTriggerReport(buf);
  }
  void FWEPWrapper::updateRollingReport()
  {
    trh_.packedTriggerReportToTable();
    if(rollingLsIndex_==0){rollingLsIndex_=lsRollSize_; rollingLsWrap_ = true;}
    rollingLsIndex_--;
    xdata::UnsignedInteger32* lsp = 0;
    xdata::UnsignedInteger32* psp = 0;
    TriggerReportStatic *tr = trh_.getPackedTriggerReportAsStruct();
    lsTriplet lst;
    lst.ls = tr->lumiSection;
    lsid_ = tr->lumiSection;
    lst.proc = tr->eventSummary.totalEvents;
    lst.acc = tr->eventSummary.totalEventsPassed;
    xdata::Serializable *psid = 0;
    xdata::Serializable *lsid = 0;
    xdata::Serializable *nbs = 0;
    xdata::Serializable *nbsr = 0;
    try{
      lsid =applicationInfoSpace_->find("lumiSectionIndex");
      if(lsid!=0){
	lsp = ((xdata::UnsignedInteger32*)lsid); 
	lsp->value_= tr->lumiSection;
      }
      psid = applicationInfoSpace_->find("lastLumiPrescaleIndex");
      if(psid!=0) {
	psp = ((xdata::UnsignedInteger32*)psid);
	if(tr->eventSummary.totalEvents != 0)
	  psp->value_ = tr->prescaleIndex;
      }
      nbs  = applicationInfoSpace_->find("nbSubProcesses");
      nbsr = applicationInfoSpace_->find("nbSubProcessesReporting");
    }
    catch(xdata::exception::Exception e){
    }

    xdata::Table::iterator it = scalersComplete_.begin();
    if(lsp)
      it->setField("lsid", *lsp);
    if(psp)
      it->setField("psid", *psp);
    if(nbs)
      it->setField("exprep", *nbs);
    else
      std::cout << "nbSubProcesses item not found !!!" << std::endl;
    if(nbsr)
      it->setField("effrep", *nbsr);
    else
      std::cout << "nbSubProcessesReporting item not found !!!" << std::endl;

    it->setField("proc",trh_.getProcThisLumi());
    it->setField("acc",trh_.getAccThisLumi());
    it->setField("triggerReport",trh_.getTableWithNames());
    lumiSectionsCtr_[rollingLsIndex_] = lst;

  }


  void FWEPWrapper::createAndSendScalersMessage()
  {
    toolbox::net::URL url(rcms_->getContextDescriptor()->getURL());
    toolbox::net::URL at(xappDesc_->getContextDescriptor()->getURL() + "/" + xappDesc_->getURN());
    toolbox::net::URL properurl(url.getProtocol(),url.getHost(),url.getPort(),"");
    xdaq::ContextDescriptor *ctxdsc = new xdaq::ContextDescriptor(properurl.toString());
    xdaq::ApplicationDescriptor *appdesc = new xdaq::ApplicationDescriptorImpl(ctxdsc,rcms_->getClassName(),rcms_->getLocalId(), "pippo");
    
    appdesc->setAttribute("path","/rcms/servlet/monitorreceiver");

    xoap::MessageReference msg = xoap::createMessage();
    xoap::SOAPEnvelope envelope = msg->getSOAPPart().getEnvelope();
    xoap::SOAPName responseName = envelope.createName( "report", xmas::NamespacePrefix, xmas::NamespaceUri);
    (void) envelope.getBody().addBodyElement ( responseName );		
    xoap::SOAPName reportName ("report", xmas::NamespacePrefix, xmas::NamespaceUri);
    xoap::SOAPElement reportElement = envelope.getBody().getChildElements(reportName)[0];
    reportElement.addNamespaceDeclaration (xmas::sensor::NamespacePrefix, xmas::sensor::NamespaceUri);
    xoap::SOAPName sampleName = envelope.createName( "sample", xmas::NamespacePrefix, xmas::NamespaceUri);
    xoap::SOAPElement sampleElement = reportElement.addChildElement(sampleName);
    xoap::SOAPName flashListName = envelope.createName( "flashlist", "", "");
    sampleElement.addAttribute(flashListName,"urn:xdaq-flashlist:scalers");
    xoap::SOAPName tagName = envelope.createName( "tag", "", "");
    sampleElement.addAttribute(tagName,"tag");
    xoap::MimeHeaders* headers = msg->getMimeHeaders();
    headers->removeHeader("x-xdaq-tags");
    headers->addHeader("x-xdaq-tags", "tag");
    tagName = envelope.createName( "originator", "", "");
    sampleElement.addAttribute(tagName,at.toString());

    xdata::exdr::AutoSizeOutputStreamBuffer outBuffer;
    xdata::exdr::Serializer serializer;
    try
      {
	serializer.exportAll( &scalersComplete_, &outBuffer );
      }
    catch(xdata::exception::Exception & e)
      {
	LOG4CPLUS_WARN(log_,
		       "Exception in serialization of scalers table");      
	//	localLog("-W- Exception in serialization of scalers table");      
	throw;
      }
  
    xoap::AttachmentPart * attachment = msg->createAttachmentPart(outBuffer.getBuffer(), outBuffer.tellp(), "application/x-xdata+exdr");
    attachment->setContentEncoding("binary");
    tagName = envelope.createName( "tag", "", "");
    sampleElement.addAttribute(tagName,"tag");
    std::stringstream contentId;

    contentId << "<" <<  "urn:xdaq-flashlist:scalers" << "@" << at.getHost() << ">";
    attachment->setContentId(contentId.str());
    std::stringstream contentLocation;
    contentId << at.toString();
    attachment->setContentLocation(contentLocation.str());
  
    std::stringstream disposition;
    disposition << "attachment; filename=" << "urn:xdaq-flashlist:scalers" << ".exdr; creation-date=" << "\"" << "dummy" << "\"";
    attachment->addMimeHeader("Content-Disposition",disposition.str());
    msg->addAttachmentPart(attachment);

    try{
      xappCtxt_->postSOAP(msg,*(xappDesc_),*appdesc);
    }
    catch(xdaq::exception::Exception &ex)
      {
	std::string message = "exception when posting SOAP message to MonitorReceiver";
	message += ex.what();
	LOG4CPLUS_WARN(log_,message.c_str());
	std::string lmessage = "-W- "+message;
	delete appdesc; 
	delete ctxdsc;
	throw;
	//	localLog(lmessage);
      }
    delete appdesc; 
    delete ctxdsc;
  }
}
