////////////////////////////////////////////////////////////////////////////////
//
// FUEventProcessor
// ----------------
//
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/Processor/interface/FUEventProcessor.h"

#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "EventFilter/Utilities/interface/ServiceWebRegistry.h"
#include "EventFilter/Utilities/interface/ServiceWeb.h"
#include "EventFilter/Utilities/interface/TimeProfilerService.h"
#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/Utilities/interface/ParameterSetRetriever.h"
#include "EventFilter/Utilities/interface/MicroStateService.h"
#include "EventFilter/Message2log4cplus/interface/MLlog4cplus.h"
#include "EventFilter/Modules/interface/FUShmDQMOutputService.h"


#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "xdaq/ApplicationDescriptorImpl.h"
#include "xdaq/ContextDescriptor.h"

#include "xcept/tools.h"
#include "xgi/Method.h"

#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"

#include "xoap/MessageReference.h"
#include "xoap/MessageFactory.h"
#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPBody.h"
#include "xoap/domutils.h"
#include "xoap/Method.h"
#include "xmas/xmas.h"
#include "xdata/TableIterator.h"
#include "xdata/exdr/Serializer.h"
#include "xdata/exdr/AutoSizeOutputStreamBuffer.h"

#include <typeinfo>
#include <stdlib.h>


namespace evf {

  namespace internal {
    
    using namespace std;
    void addService(vector<edm::ParameterSet>& adjust,string const& service)
    {
      edm::ParameterSet newpset;
      newpset.addParameter<string>("@service_type",service);
      adjust.push_back(newpset);
    }

    // Add a service to the services list if it is not already there
    void addServiceMaybe(vector<edm::ParameterSet>& adjust,string const& service)
    {
      std::vector<edm::ParameterSet>::const_iterator it;
      for(it=adjust.begin();it!=adjust.end();++it) {
	string name = it->getParameter<std::string>("@service_type");
	if (name == service) return;
      }
      addService(adjust, service);
    }
    
    const edm::ParameterSet *findService(vector<edm::ParameterSet> &adjust, string const& service)
    {
      edm::ParameterSet *retval = 0;
      std::vector<edm::ParameterSet>::const_iterator it;
      for(it=adjust.begin();it!=adjust.end();++it) {
	string name = it->getParameter<std::string>("@service_type");
	if (name == service) return &(*it);
      }
      return retval;
    }
    
  } // namespace internal
  
} // namespace evf


using namespace std;
using namespace evf;
using namespace cgicc;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUEventProcessor::FUEventProcessor(xdaq::ApplicationStub *s) 
  : xdaq::Application(s)
  , fsm_(this)
  , evtProcessor_(0)
  , serviceToken_()
  , servicesDone_(false)
  , inRecovery_(false)
  , recoveryCount_(0)
  , triggerReportIncomplete_(false)
  , prescaleSvc_(0)
  , runNumber_(0)
  , epInitialized_(false)
  , outPut_(true)
  , inputPrescale_(1)
  , outputPrescale_(1)
  , timeoutOnStop_(10)
  , hasShMem_(true)
  , hasPrescaleService_(true)
  , hasModuleWebRegistry_(true)
  , hasServiceWebRegistry_(true)
  , isRunNumberSetter_(true)
  , isPython_(false)
  , outprev_(true)
  , monSleepSec_(1)
  , wlMonitoring_(0)
  , asMonitoring_(0)
  , watching_(false)
  , wlScalers_(0)
  , asScalers_(0)
  , localLsIncludingTimeOuts_(0)
  , lsTimeOut_(105)
  , firstLsTimeOut_(200)
  , residualTimeOut_(0)
  , lastLsTimedOut_(false)
  , lastLsWithEvents_(0)
  , lastLsWithTimeOut_(0)
  , reasonForFailedState_()
  , wlMonitoringActive_(false)
  , wlScalersActive_(false)
  , scalersUpdateAttempted_(0)
  , scalersUpdateCounter_(0)
  , lumiSectionsCtr_(lsRollSize_)
  , lumiSectionsTo_(lsRollSize_)
  , allPastLumiProcessed_(0)
  , rollingLsIndex_(lsRollSize_)
  , rollingLsWrap_(false)
  , squidnet_(3128,"http://localhost:8000/RELEASE-NOTES.txt")
  , logRing_(logRingSize_)
  , logRingIndex_(logRingSize_)
  , logWrap_(false)
{
  //list of variables for scalers flashlist
  names_.push_back("lumiSectionIndex");
  names_.push_back("prescaleSetIndex");
  names_.push_back("scalersTable");
  squidPresent_ = squidnet_.check();

  // bind relevant callbacks to finite state machine
  fsm_.initialize<evf::FUEventProcessor>(this);
  
  //set sourceId_
  url_ =
    getApplicationDescriptor()->getContextDescriptor()->getURL()+"/"+
    getApplicationDescriptor()->getURN();
  class_   =getApplicationDescriptor()->getClassName();
  instance_=getApplicationDescriptor()->getInstance();
  sourceId_=class_.toString()+"_"+instance_.toString();
  LOG4CPLUS_INFO(getApplicationLogger(),sourceId_ <<" constructor");
  LOG4CPLUS_INFO(getApplicationLogger(),"plugin path:"<<getenv("SEAL_PLUGINS"));
  LOG4CPLUS_INFO(getApplicationLogger(),"CMSSW_BASE:"<<getenv("CMSSW_BASE"));
  
  getApplicationDescriptor()->setAttribute("icon", "/evf/images/epicon.jpg");

  ostringstream ns;  ns << "EP" << instance_.toString();


  //some initialization of state data
  epMAltState_ = -1;
  epmAltState_ = -1;
  
  xdata::InfoSpace *ispace = getApplicationInfoSpace();

  // default configuration
  ispace->fireItemAvailable("parameterSet",         &configString_);
  ispace->fireItemAvailable("pluginPath",           &sealPluginPath_);
  ispace->fireItemAvailable("epInitialized",        &epInitialized_);
  ispace->fireItemAvailable("stateName",             fsm_.stateName());
  ispace->fireItemAvailable("runNumber",            &runNumber_);
  ispace->fireItemAvailable("outputEnabled",        &outPut_);
  ispace->fireItemAvailable("globalInputPrescale",  &inputPrescale_);
  ispace->fireItemAvailable("globalOutputPrescale", &outputPrescale_);
  ispace->fireItemAvailable("timeoutOnStop",        &timeoutOnStop_);
  ispace->fireItemAvailable("hasSharedMemory",      &hasShMem_);
  ispace->fireItemAvailable("hasPrescaleService",   &hasPrescaleService_);
  ispace->fireItemAvailable("hasModuleWebRegistry", &hasModuleWebRegistry_);
  ispace->fireItemAvailable("hasServiceWebRegistry", &hasServiceWebRegistry_);
  ispace->fireItemAvailable("isRunNumberSetter",    &isRunNumberSetter_);
  ispace->fireItemAvailable("isPython",             &isPython_);
  ispace->fireItemAvailable("monSleepSec",          &monSleepSec_);
  ispace->fireItemAvailable("lsTimeOut",            &lsTimeOut_);
  ispace->fireItemAvailable("rcmsStateListener",     fsm_.rcmsStateListener());
  ispace->fireItemAvailable("foundRcmsStateListener",fsm_.foundRcmsStateListener());
  
  
  ispace->fireItemAvailable("prescalerAsString",    &prescalerAsString_);
  //  ispace->fireItemAvailable("triggerReportAsString",&triggerReportAsString_);
  
  // Add infospace listeners for exporting data values
  getApplicationInfoSpace()->addItemChangedListener("parameterSet",        this);
  getApplicationInfoSpace()->addItemChangedListener("outputEnabled",       this);
  getApplicationInfoSpace()->addItemChangedListener("globalInputPrescale", this);
  getApplicationInfoSpace()->addItemChangedListener("globalOutputPrescale",this);

  // findRcmsStateListener
  fsm_.findRcmsStateListener();
  
  // initialize monitoring infospace

  std::stringstream oss2;
  oss2<<"urn:xdaq-monitorable-"<<class_.toString();
  string monInfoSpaceName=oss2.str();
  toolbox::net::URN urn = this->createQualifiedInfoSpace(monInfoSpaceName);
  monitorInfoSpace_ = xdata::getInfoSpaceFactory()->get(urn.toString());

  
  monitorInfoSpace_->fireItemAvailable("url",                      &url_);
  monitorInfoSpace_->fireItemAvailable("class",                    &class_);
  monitorInfoSpace_->fireItemAvailable("instance",                 &instance_);
  monitorInfoSpace_->fireItemAvailable("runNumber",                &runNumber_);
  monitorInfoSpace_->fireItemAvailable("stateName",                 fsm_.stateName()); 
  monitorInfoSpace_->fireItemAvailable("epMacroState",             &epMState_);
  monitorInfoSpace_->fireItemAvailable("epMicroState",             &epmState_);
  monitorInfoSpace_->fireItemAvailable("nbProcessed",              &nbProcessed_);
  monitorInfoSpace_->fireItemAvailable("nbAccepted",               &nbAccepted_);

  monitorInfoSpace_->fireItemAvailable("epMacroStateInt",             &epMAltState_);
  monitorInfoSpace_->fireItemAvailable("epMicroStateInt",             &epmAltState_);
  
  monitorInfoSpace_->fireItemAvailable("macroStateLegend",      &macro_state_legend_);
  monitorInfoSpace_->fireItemAvailable("microStateLegend",      &micro_state_legend_);

  monitorInfoSpace_->fireItemAvailable("squidPresent",      &squidPresent_);

  std::stringstream oss3;
  oss3<<"urn:xdaq-scalers-"<<class_.toString();
  string monInfoSpaceName2=oss3.str();
  toolbox::net::URN urn2 = this->createQualifiedInfoSpace(monInfoSpaceName2);
  xdata::Table &stbl = trh_.getTable(); 
  scalersInfoSpace_ = xdata::getInfoSpaceFactory()->get(urn2.toString());
  scalersInfoSpace_->fireItemAvailable("instance", &instance_);
  scalersInfoSpace_->fireItemAvailable("scalersTable", &stbl);
  scalersComplete_.addColumn("instance", "unsigned int 32");
  scalersComplete_.addColumn("lsid", "unsigned int 32");
  scalersComplete_.addColumn("psid", "unsigned int 32");
  scalersComplete_.addColumn("triggerReport", "table");  

  // bind prescale related soap callbacks
  //  xoap::bind(this,&FUEventProcessor::getPsReport ,"GetPsReport",XDAQ_NS_URI);
  //  xoap::bind(this,&FUEventProcessor::getLsReport ,"GetLsReport",XDAQ_NS_URI);
  //  xoap::bind(this,&FUEventProcessor::putPrescaler,"PutPrescaler",XDAQ_NS_URI);
  
  // Bind web interface
  xgi::bind(this, &FUEventProcessor::css           ,   "styles.css");
  xgi::bind(this, &FUEventProcessor::defaultWebPage,   "Default"   );
  xgi::bind(this, &FUEventProcessor::spotlightWebPage, "Spotlight" );
  xgi::bind(this, &FUEventProcessor::moduleWeb     ,   "moduleWeb" );
  xgi::bind(this, &FUEventProcessor::serviceWeb    ,   "serviceWeb" );
  xgi::bind(this, &FUEventProcessor::microState    ,   "microState");

  // instantiate the plugin manager, not referenced here after!

  edm::AssertHandler ah;

  try{
    LOG4CPLUS_DEBUG(getApplicationLogger(),
		    "Trying to create message service presence ");
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


//______________________________________________________________________________
FUEventProcessor::~FUEventProcessor()
{
  if (evtProcessor_) delete evtProcessor_;
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
bool FUEventProcessor::getTriggerReport(bool useLock)
  throw (toolbox::fsm::exception::Exception)
{


  // Calling this method results in calling 
  // evtProcessor_->getTriggerReport, the value returned is encoded as
  // a xdata::Table.
  LOG4CPLUS_DEBUG(getApplicationLogger(),"getTriggerReport action invoked");
  if(inRecovery_) { return false;} //stop scalers loop if caught in the middle of recovery

  //Get the trigger report.
  ModuleWebRegistry *mwr = 0;
  try{
    if(edm::Service<ModuleWebRegistry>().isAvailable())
      mwr = edm::Service<ModuleWebRegistry>().operator->();
  }
  catch(...) { 
    LOG4CPLUS_INFO(getApplicationLogger(),
		   "exception when trying to get service ModuleWebRegistry");
    return false;
  }
  edm::TriggerReport tr; 
  if(mwr==0) return false;

  xdata::InfoSpace *ispace = getApplicationInfoSpace();
  unsigned int ls = 0;
  unsigned int ps = 0;
  xdata::Table::iterator it = scalersComplete_.begin();
  if( it == scalersComplete_.end())
    {
      it = scalersComplete_.append();
      it->setField("instance",instance_);
    }
  timeval tv;
  if(useLock) {
    gettimeofday(&tv,0);
    mwr->openBackDoor("DaqSource",residualTimeOut_);
    string st = fsm_.stateName()->toString();
    if(st!="Enabled" && st!="Configured" && st!="enabling" && st!="stopping") return false;
    residualTimeOut_ = lsTimeOut_.value_ ;
  }
  bool localTimeOut = false;
  try{
    xdata::Serializable *lsid = ispace->find("lumiSectionIndex");
    if(lsid) {
      ls = ((xdata::UnsignedInteger32*)(lsid))->value_;

      xdata::Boolean *to =  (xdata::Boolean*)ispace->find("lsTimedOut");
      if(to!=0)
	{
	  localTimeOut = to->value_;
	  if(to->value_)
	    {
	      if(lastLsTimedOut_)localLsIncludingTimeOuts_.value_++;
	      else localLsIncludingTimeOuts_.value_ = ls;
	      lastLsTimedOut_ = true; 
	      lastLsWithTimeOut_ = ls;
	    }
	  else
	    {
	      lastLsWithEvents_ = ls;
	      if(lastLsTimedOut_)
		{
		  if(localLsIncludingTimeOuts_.value_ < (ls-1)) //cover timed out LS not yet accounted for when events return;
		    for(localLsIncludingTimeOuts_.value_++; localLsIncludingTimeOuts_.value_ < ls; localLsIncludingTimeOuts_.value_++)
		      {
			if(rollingLsIndex_==0){rollingLsIndex_=lsRollSize_; rollingLsWrap_ = true;}
			rollingLsIndex_--;
			lumiSectionsTo_[rollingLsIndex_] = localTimeOut;
			lumiSectionsCtr_[rollingLsIndex_] = pair<unsigned int, unsigned int>(localLsIncludingTimeOuts_.value_,
											     evtProcessor_->totalEvents()-
											     allPastLumiProcessed_);
			it->setField("lsid", localLsIncludingTimeOuts_);
			fireScalersUpdate();
		      }

		  timeval tv1;
		  gettimeofday(&tv1,0);
		  residualTimeOut_ -= (tv1.tv_sec - tv.tv_sec); //adjust timeout to handle rest of LS where events come back
		  mwr->closeBackDoor("DaqSource"); 
		  lastLsTimedOut_ = false;
		  return true;
		}
	      localLsIncludingTimeOuts_.value_ = ls;
	      lastLsTimedOut_ = false; 
	    }
	}
      it->setField("lsid", localLsIncludingTimeOuts_);
    }
    xdata::Serializable *psid = ispace->find("prescaleSetIndex");
    if(psid) {
      ps = ((xdata::UnsignedInteger32*)(psid))->value_;
      if(prescaleSvc_ != 0) prescaleSvc_->setIndex(ps);
      it->setField("psid",*psid);
    }
  }
  catch(xdata::exception::Exception e){
    LOG4CPLUS_INFO(getApplicationLogger(),
                   "exception when obtaining ls or ps id");
    if(useLock){
      mwr->closeBackDoor("DaqSource");
    }
    return false;
  }


  if(rollingLsIndex_==0){rollingLsIndex_=lsRollSize_; rollingLsWrap_ = true;}
  rollingLsIndex_--;
  lumiSectionsTo_[rollingLsIndex_] = localTimeOut;
  lumiSectionsCtr_[rollingLsIndex_] = pair<unsigned int, unsigned int>(localLsIncludingTimeOuts_.value_,
								       evtProcessor_->totalEvents()-
								       allPastLumiProcessed_);
  allPastLumiProcessed_ = evtProcessor_->totalEvents();


  if(!inRecovery_)evtProcessor_->getTriggerReport(tr);
  if(useLock){
    mwr->closeBackDoor("DaqSource");
  }  

  trh_.formatReportTable(tr,descs_);
  if(trh_.checkLumiSection(ls))
    {
      trh_.triggerReportToTable(tr,ls,false);
    }
  else
    {
      if(triggerReportIncomplete_)
	{
	  triggerReportIncomplete_ = false;
	  //	      trh_.printReportTable();
	  //send xmas message with data
	}
      trh_.triggerReportToTable(tr,ls);
    }
  it->setField("triggerReport",trh_.getTable());
  // send xmas message with data
  //      triggerReportAsString_ = triggerReportToString(tr);
      
  //Print the trigger report message in debug format.
  //      trh_.printTriggerReport(tr);
  
  
  return true;
}



//______________________________________________________________________________
bool FUEventProcessor::configuring(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start configuring ...");
    initEventProcessor();
    if(epInitialized_)
      {
	startMonitoringWorkLoop();
	evtProcessor_->beginJob();
	fsm_.fireEvent("ConfigureDone",this);
	LOG4CPLUS_INFO(getApplicationLogger(),"Finished configuring!");
      }
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "configuring FAILED: " + (string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
  }
  localLog("-I- Configuration completed");
  return false;
}


//______________________________________________________________________________
bool FUEventProcessor::enabling(toolbox::task::WorkLoop* wl)
{
  unsigned int tempLsTO = lsTimeOut_.value_;
  lsTimeOut_.value_ = firstLsTimeOut_;
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start enabling ...");
    
    // if the ep is intialized already, the initialization will be skipped
    initEventProcessor();
    if(hasShMem_) attachDqmToShm();

    int sc = 0;
    evtProcessor_->clearCounters();
    if(isRunNumberSetter_)
      evtProcessor_->setRunNumber(runNumber_.value_);
    else
      evtProcessor_->declareRunNumber(runNumber_.value_);
    try{
      evtProcessor_->runAsync();
      sc = evtProcessor_->statusAsync();
    }
    catch(cms::Exception &e) {
      reasonForFailedState_ = e.explainSelf();
      fsm_.fireFailed(reasonForFailedState_,this);
      return false;
    }    
    catch(std::exception &e) {
      reasonForFailedState_  = e.what();
      fsm_.fireFailed(reasonForFailedState_,this);
      return false;
    }
    catch(...) {
      reasonForFailedState_ = "Unknown Exception";
      fsm_.fireFailed(reasonForFailedState_,this);
      return false;
    }
    
    if(sc != 0) {
      ostringstream oss;
      oss<<"EventProcessor::runAsync returned status code " << sc;
      reasonForFailedState_ = oss.str();
      fsm_.fireFailed(reasonForFailedState_,this);
      return false;
    }
    
    LOG4CPLUS_INFO(getApplicationLogger(),"Finished enabling!");
    
    fsm_.fireEvent("EnableDone",this);
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "enabling FAILED: " + (string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
  }
  while(evtProcessor_->getState()!= edm::event_processor::sRunning){
    LOG4CPLUS_INFO(getApplicationLogger(),"waiting for edm::EventProcessor to start before enabling watchdog");
    ::sleep(1);
  }
  watching_ = true;
  residualTimeOut_ = lsTimeOut_.value_;
  startScalersWorkLoop();
  lsTimeOut_.value_ = tempLsTO;
  localLog("-I- Start completed");
  return false;
}


//______________________________________________________________________________
bool FUEventProcessor::stopping(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start stopping :) ...");
    edm::EventProcessor::StatusCode rc = stopEventProcessor();
    LOG4CPLUS_INFO(getApplicationLogger(),"Finished stopping!");
    if(rc != edm::EventProcessor::epTimedOut) 
      fsm_.fireEvent("StopDone",this);
    else
      {
	epMState_ = evtProcessor_->currentStateName();
	reasonForFailedState_ = "EventProcessor stop timed out";
	localLog(reasonForFailedState_);
	fsm_.fireFailed(reasonForFailedState_,this);

      }
    if(hasShMem_) detachDqmFromShm();
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "stopping FAILED: " + (string)e.what();
    localLog(reasonForFailedState_);
    fsm_.fireFailed(reasonForFailedState_,this);
  }
  watching_ = false;
  allPastLumiProcessed_ = 0;
  localLog("-I- Stop completed");
  return false;
}


//______________________________________________________________________________
bool FUEventProcessor::halting(toolbox::task::WorkLoop* wl)
{
  edm::ServiceRegistry::Operate operate(serviceToken_);
  ModuleWebRegistry *mwr = 0;
  try{
    if(edm::Service<ModuleWebRegistry>().isAvailable())
      mwr = edm::Service<ModuleWebRegistry>().operator->();
  }
  catch(...) { 
    LOG4CPLUS_INFO(getApplicationLogger(),
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
    LOG4CPLUS_INFO(getApplicationLogger(),
		   "exception when trying to get service ModuleWebRegistry");
  }
  
  if(swr) 
    {
      swr->clear();
    }

  edm::event_processor::State st = evtProcessor_->getState();
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start halting ...");
    edm::EventProcessor::StatusCode rc = stopEventProcessor();
    if(rc != edm::EventProcessor::epTimedOut)
      {
	if(hasShMem_) detachDqmFromShm();
	if(st == edm::event_processor::sJobReady || st == edm::event_processor::sDone)
	  evtProcessor_->endJob();
	monitorInfoSpace_->lock(); //protect monitoring workloop from using ep pointer while it is being deleted
	delete evtProcessor_;
	evtProcessor_ = 0;
	monitorInfoSpace_->unlock();
	epInitialized_ = false;
	LOG4CPLUS_INFO(getApplicationLogger(),"Finished halting!");
  
	fsm_.fireEvent("HaltDone",this);
      }
    else
      {
	reasonForFailedState_ = "EventProcessor stop timed out";
	localLog(reasonForFailedState_);
	fsm_.fireFailed(reasonForFailedState_,this);
      }
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "halting FAILED: " + (string)e.what();
    localLog(reasonForFailedState_);
    fsm_.fireFailed(reasonForFailedState_,this);
  }
  watching_ = false;
  allPastLumiProcessed_ = 0;
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
void FUEventProcessor::initEventProcessor()
{
  trh_.resetFormat(); //reset the report table even if HLT didn't change
  if (epInitialized_) {
    LOG4CPLUS_INFO(getApplicationLogger(),
		   "CMSSW EventProcessor already initialized: skip!");
    return;
  }
  
  LOG4CPLUS_INFO(getApplicationLogger(),"Initialize CMSSW EventProcessor.");
  
  if (0!=setenv("SEAL_PLUGINS",sealPluginPath_.value_.c_str(),0)) {
    LOG4CPLUS_ERROR(getApplicationLogger(),"Failed to set SEAL_PLUGINS search path");
  }
  else {
    LOG4CPLUS_INFO(getApplicationLogger(),"plugin path: "<<getenv("SEAL_PLUGINS"));
    LOG4CPLUS_INFO(getApplicationLogger(),"CMSSW_BASE:"<<getenv("CMSSW_BASE"));
  }
  

  // job configuration string
  ParameterSetRetriever pr(configString_.value_);
  configuration_ = pr.getAsString();
  if (configString_.value_.size() > 3 && configString_.value_.substr(configString_.value_.size()-3) == ".py") isPython_ = true;
  boost::shared_ptr<edm::ParameterSet> params; // change this name!
  boost::shared_ptr<vector<edm::ParameterSet> > pServiceSets;
  boost::shared_ptr<edm::ProcessDesc> pdesc;
  try{
    if(isPython_)
      {
	PythonProcessDesc ppdesc = PythonProcessDesc(configuration_);
	pdesc = ppdesc.processDesc();
      }
    else
      pdesc = boost::shared_ptr<edm::ProcessDesc>(new edm::ProcessDesc(configuration_));
  }
  catch(cms::Exception &e){
    reasonForFailedState_ = e.explainSelf();
    fsm_.fireFailed(reasonForFailedState_,this);
    return;
  } 
  pServiceSets = pdesc->getServicesPSets();
  // add default set of services
  if(!servicesDone_) {
    internal::addServiceMaybe(*pServiceSets,"DQMStore");
    //    internal::addServiceMaybe(*pServiceSets,"MonitorDaemon");
    internal::addServiceMaybe(*pServiceSets,"MLlog4cplus");
    internal::addServiceMaybe(*pServiceSets,"MicroStateService");
    if(hasPrescaleService_) internal::addServiceMaybe(*pServiceSets,"PrescaleService");
    if(hasModuleWebRegistry_) internal::addServiceMaybe(*pServiceSets,"ModuleWebRegistry");
    if(hasServiceWebRegistry_) internal::addServiceMaybe(*pServiceSets,"ServiceWebRegistry");
    
    try{
      serviceToken_ = edm::ServiceRegistry::createSet(*pServiceSets);
    }
    catch(cms::Exception &e) {
      LOG4CPLUS_ERROR(getApplicationLogger(),e.explainSelf());
    }    
    catch(std::exception &e) {
      LOG4CPLUS_ERROR(getApplicationLogger(),e.what());
    }
    catch(...) {
      LOG4CPLUS_ERROR(getApplicationLogger(),"Unknown Exception");
    }
    servicesDone_ = true;
  }
  
  edm::ServiceRegistry::Operate operate(serviceToken_);


  //test rerouting of fwk logging to log4cplus
  edm::LogInfo("FUEventProcessor")<<"started MessageLogger Service.";
  edm::LogInfo("FUEventProcessor")<<"Using config string \n"<<configuration_;

  DQMStore *dqm = 0;
  try{
    if(edm::Service<DQMStore>().isAvailable())
      dqm = edm::Service<DQMStore>().operator->();
  }
  catch(...) { 
    LOG4CPLUS_INFO(getApplicationLogger(),
		   "exception when trying to get service DQMStore");
  }
  if(dqm!=0) dqm->rmdir("");
  

  ModuleWebRegistry *mwr = 0;
  try{
    if(edm::Service<ModuleWebRegistry>().isAvailable())
      mwr = edm::Service<ModuleWebRegistry>().operator->();
  }
  catch(...) { 
    LOG4CPLUS_INFO(getApplicationLogger(),
		   "exception when trying to get service ModuleWebRegistry");
  }

  if(mwr) mwr->clear(); // in case we are coming from stop we need to clear the mwr

  ServiceWebRegistry *swr = 0;
  try{
    if(edm::Service<ServiceWebRegistry>().isAvailable())
      swr = edm::Service<ServiceWebRegistry>().operator->();
  }
  catch(...) { 
    LOG4CPLUS_INFO(getApplicationLogger(),
		   "exception when trying to get service ModuleWebRegistry");
  }

  //  if(swr) swr->clear(); // in case we are coming from stop we need to clear the swr


  // instantiate the event processor
  try{
    vector<string> defaultServices;
    vector<string> forcedServices;
    defaultServices.push_back("MessageLogger");
    defaultServices.push_back("InitRootHandlers");
    defaultServices.push_back("JobReportService");
    pdesc->addServices(defaultServices, forcedServices);
    monitorInfoSpace_->lock();
    if (0!=evtProcessor_) delete evtProcessor_;

    evtProcessor_ = new edm::EventProcessor(pdesc,
					    serviceToken_,
					    edm::serviceregistry::kTokenOverrides);

    monitorInfoSpace_->unlock();
    //    evtProcessor_->setRunNumber(runNumber_.value_);

    if(!outPut_)
      //evtProcessor_->toggleOutput();
      //evtProcessor_->prescaleInput(inputPrescale_);
      //evtProcessor_->prescaleOutput(outputPrescale_);
      evtProcessor_->enableEndPaths(outPut_);
    
    outprev_=outPut_;
    
    // to publish all module names to XDAQ infospace

    if(mwr) 
      {
	mwr->publish(getApplicationInfoSpace());
	mwr->publishToXmas(scalersInfoSpace_);
      }
    if(swr) 
      {
	swr->publish(getApplicationInfoSpace());
      }
    // get the prescale service
    LOG4CPLUS_INFO(getApplicationLogger(),
		   "Checking for edm::service::PrescaleService!");
    try {
      if(edm::Service<edm::service::PrescaleService>().isAvailable())
	{
	  LOG4CPLUS_INFO(getApplicationLogger(),
			 "edm::service::PrescaleService is available!");
	  prescaleSvc_ = edm::Service<edm::service::PrescaleService>().operator->();
	  LOG4CPLUS_INFO(getApplicationLogger(),
			 "Obtained pointer to PrescaleService");
	  //prescaleSvc_->putHandle(evtProcessor_);
	  //LOG4CPLUS_INFO(getApplicationLogger(),
	  //	 "PrescaleService::putHandle called");
	}
    }
    catch(...) {
      LOG4CPLUS_INFO(getApplicationLogger(),
		     "exception when trying to get service "
		     <<"edm::service::PrescaleService");
    }
    const edm::ParameterSet *prescaleSvcConfig = internal::findService(*pServiceSets,"PrescaleService");
    if(prescaleSvc_ != 0 && prescaleSvcConfig !=0) prescaleSvc_->reconfigure(*prescaleSvcConfig);
  }
  catch(cms::Exception &e) {
    reasonForFailedState_ = e.explainSelf();
    fsm_.fireFailed(reasonForFailedState_,this);
    return;
  }    
  catch(std::exception &e) {
    reasonForFailedState_ = e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    return;
  }
  catch(...) {
    fsm_.fireFailed("Unknown Exception",this);
    return;
  }
  
  //fill macrostate legend information
  unsigned int i = 0;
  std::stringstream oss;
  for(i = (unsigned int)edm::event_processor::sInit; i < (unsigned int)edm::event_processor::sInvalid; i++)
    {
      oss << i << "=" << evtProcessor_->stateName((edm::event_processor::State) i) << " ";
    }
  monitorInfoSpace_->lock();
  if(getApplicationDescriptor()->getInstance() == 0) macro_state_legend_ = oss.str();
  //fill microstate legend information
  descs_ = evtProcessor_->getAllModuleDescriptions();

  std::stringstream oss2;
  unsigned int outcount = 0;
  oss2 << 0 << "=In ";
  modmap_["IN"]=0;
  for(unsigned int j = 0; j < descs_.size(); j++)
    {
      if(descs_[j]->moduleName() == "ShmStreamConsumer") // find something better than hardcoding name
	{ 
	  outcount++;
	  oss2 << outcount << "=" << descs_[j]->moduleLabel() << " ";
	  modmap_[descs_[j]->moduleLabel()]=outcount;
	  i++;
	}
    }
  modmap_["DQM"]=outcount+1;
  oss2 << outcount+1 << "=DQMHistograms ";
  unsigned int modcount = 1;
  for(i = 0; i < descs_.size(); i++)
    {
      if(descs_[i]->moduleName() != "ShmStreamConsumer")
	{
	  modcount++;
	  oss2 << outcount+modcount << "=" << descs_[i]->moduleLabel() << " ";
	  modmap_[descs_[i]->moduleLabel()]=outcount+modcount;
	}
    }
  if(getApplicationDescriptor()->getInstance() == 0) micro_state_legend_ = oss2.str();
  monitorInfoSpace_->unlock();
  LOG4CPLUS_INFO(getApplicationLogger(),"FUEventProcessor configuration finished.");
  
  epInitialized_ = true;

  return;
}


//______________________________________________________________________________
edm::EventProcessor::StatusCode FUEventProcessor::stopEventProcessor()
{
  edm::event_processor::State st = evtProcessor_->getState();

  LOG4CPLUS_INFO(getApplicationLogger(),"FUEventProcessor::stopEventProcessor. state "
               << evtProcessor_->stateName(st));

  edm::EventProcessor::StatusCode rc = edm::EventProcessor::epSuccess;

  if(st == edm::event_processor::sInit) return rc;

  try  {
    rc = evtProcessor_->waitTillDoneAsync(timeoutOnStop_.value_);
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
  return rc;

}


//______________________________________________________________________________
void FUEventProcessor::actionPerformed(xdata::Event& e)
{
  if (e.type()=="ItemChangedEvent" && !(fsm_.stateName()->toString()=="Halted")) {
    string item = dynamic_cast<xdata::ItemChangedEvent&>(e).itemName();
    
    if ( item == "parameterSet") {
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
      //evtProcessor_->prescaleInput(inputPrescale_);
      //LOG4CPLUS_WARN(this->getApplicationLogger(),
      //"Setting global input prescale factor to" << inputPrescale_);
      //
      LOG4CPLUS_WARN(getApplicationLogger(),
		     "Setting global input prescale has no effect "
		     <<"in this version of the code");
    }
    if ( item == "globalOutputPrescale") {
      //evtProcessor_->prescaleOutput(outputPrescale_);
      //LOG4CPLUS_WARN(this->getApplicationLogger(),
      //"Setting global output prescale factor to" << outputPrescale_);
      LOG4CPLUS_WARN(getApplicationLogger(),
		     "Setting global output prescale has no effect "
		     <<"in this version of the code");
    }
  }

}

//______________________________________________________________________________

//______________________________________________________________________________
void FUEventProcessor::defaultWebPage(xgi::Input  *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{


  string urn = getApplicationDescriptor()->getURN();
  *out << "<!-- base href=\"/" <<  urn
       << "\"> -->" << endl;
  *out << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">"	<< endl;
  *out << "<html>"								<< endl;
  *out << "<head>"								<< endl;

  *out << "<script src=\"/evf/html/microEPPage.js\"></script>"<< endl;
  *out << "<style type=\"text/css\">"						<< endl;
  *out << "#s1 {"								<< endl;
  *out << "border-width: 2px; border: solid blue; text-align: right; "
       << "background: lightgrey "						<< endl;
  *out << "}"									<< endl; 
  *out << "#s2 {"								<< endl;
  *out << "border-width: 2px; border: white; text-align: left; vertical-align: top; "
       << "background: green; font-size: 12pt; height:112; width:80 "		<< endl;
  *out << "}"									<< endl; 
  *out << "</style> "								<< endl; 
  *out << "<link type=\"text/css\" rel=\"stylesheet\""
       << " href=\"/" <<  urn
       << "/styles.css\"/>"							<< endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName() 
       << " " << getApplicationDescriptor()->getInstance() 
       << " MAIN</title>"							<< endl;
  *out << "</head>"								<< endl;
  *out << "<body onload=\"loadXMLDoc()\">"					<< endl;

  *out << "<table border=\"0\" width=\"100%\">"					<< endl;

  *out << "<tr>"								<< endl;

  *out << "  <td align=\"left\"><img align=\"middle\" src=\"/evf/images/epicon.jpg\""
       << " alt=\"main\" width=\"64\" height=\"64\"></td>"                      << endl;

  *out << "  <td align=\"middle\">"						<< endl;
  *out << "    <table><tr><td>"							<< endl;
  *out << "      <b>" 
       << getApplicationDescriptor()->getClassName() << " " 
       << getApplicationDescriptor()->getInstance() 
       << "</b></td></tr>"			<< endl;
  *out << "      <tr><td>Run Number: " << runNumber_.toString() << "</td></tr>"	<< endl;
  if(fsm_.stateName()->toString() != "Halted" && fsm_.stateName()->toString() != "halting")
    *out << "      <tr><td><a href=\"" << configString_.toString() << "\">HLT Config</a></td></tr>"	<< endl;
  *out << "    </table>"							<< endl;
  *out << "  </td>"								<< endl;

  *out << "  <td align=\"middle\">"						<< endl;
  *out << "    <table><tr>"							<< endl;
  *out << "       <td><div id=\"s2\">"
       << "</div></td>"			<< endl;
  *out << "       <td><div id=\"s1\" style=\"border:2px";
  if(fsm_.stateName()->value_ == "Failed")
    {
      *out << " solid red;height:64;width:150\">microState</div> "		<< endl;
      *out << "    </td></tr><tr><td>"						<< endl;
      *out << "                <textarea rows=" << 5 << " cols=50 scroll=yes";
      *out << " readonly title=\"Reason For Failed\">" << reasonForFailedState_;
      *out << "</textarea></td></tr></table>"					<< endl;
    }
  else
    {
      *out << ";height:112;width:150\">microState</div> "			<< endl;
      *out << "     </td></tr></table>"						<< endl;
    }
  *out << "  </td>"								<< endl;


  *out << "  <td width=\"32\"><a href=\"/urn:xdaq-application:lid=3\">"
       << "<img align=\"middle\" src=\"/hyperdaq/images/HyperDAQ.jpg\" alt=\"HyperDAQ\""
       << " width=\"32\" height=\"32\"></a></td>"                               << endl;

  *out << "  <td width=\"32\"><a href=\"/" << urn 
       << "/Spotlight\"><img align=\"middle\" src=\"/evf/images/spoticon.jpg\"";
  *out << " alt=\"debug\" width=\"32\" height=\"32\"></a></td>"                 << endl;

  *out << "</tr>"								<< endl;
  //version number, please update consistently with TAG
  *out << "<tr>"								<< endl;
  *out << "  <td colspan=\"5\" align=\"right\">"				<< endl;
  *out << "    Version 1.4.7"							<< endl;
  *out << "  </td>"								<< endl;
  *out << "</tr>"								<< endl;

  *out << "</table>"								<< endl;
  // end of page banner table
  *out << "<hr>"								<< endl;

  *out << "<table>"                                                  << endl;
  *out << "<tr valign=\"top\">"                                      << endl;
  *out << "  <td>"                                                   << endl;

  //configuration table

  *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">" << endl;
  *out << "<colgroup> <colgroup align=\"right\">"                    << endl;
  *out << "  <tr>"                                                   << endl;
  *out << "    <th colspan=2>"                                       << endl;
  *out << "      " << "Configuration"                                << endl;
  *out << "    </th>"                                                << endl;
  *out << "  </tr>"                                                  << endl;
  
  *out << "<tr>" << endl;
  *out << "<th >" << endl;
  *out << "Parameter" << endl;
  *out << "</th>" << endl;
  *out << "<th>" << endl;
  *out << "Value" << endl;
  *out << "</th>" << endl;
  *out << "</tr>" << endl;

  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Output Enabled" << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << outPut_.toString() << endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Timeout On Stop" << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << timeoutOnStop_.toString() << endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Has Shared Memory" << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << hasShMem_.toString() << endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Is Python Config" << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << isPython_.toString() << endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Has Module Web" << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << hasModuleWebRegistry_.toString() << endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Has Service Web" << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << hasServiceWebRegistry_.toString() << endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Monitor Sleep (s)" << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << monSleepSec_.toString() << endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "LumiSec Timeout (s)" << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << lsTimeOut_.toString() << endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  *out << "</table>"							<< endl;
  

  *out << "<td>"							<< endl;

  //status table

  *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">"	<< endl;
  *out << "<colgroup> <colgroup align=\"right\">"			<< endl;
  *out << "  <tr>"							<< endl;
  *out << "    <th colspan=2>"						<< endl;
  *out << "      " << "Status"						<< endl;
  *out << "    </th>"							<< endl;
  *out << "  </tr>"							<< endl;
  *out << "  <tr>"							<< endl;
  *out << "    <th >"							<< endl;
  *out << "       Parameter"						<< endl;
  *out << "    </th>"							<< endl;
  *out << "    <th>"							<< endl;
  *out << "       Value"						<< endl;
  *out << "    </th>"							<< endl;
  *out << "  </tr>"							<< endl;
  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Successful Recoveries" << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << recoveryCount_ << endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;

  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Squid Present " << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << squidPresent_.toString() << endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Monitor WL " << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  if(wlMonitoring_!=0 && wlMonitoring_->isActive()) *out << (wlMonitoringActive_ ? "active" : "inactive");
  else *out << "not initialized";
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Scalers WL " << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  if(wlScalers_!=0 && wlScalers_->isActive()) *out << (wlScalersActive_ ? "active" : "inactive");
  else *out << "not initialized";
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Scalers Updates (Att/Succ)" << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << scalersUpdateAttempted_ << "/" << scalersUpdateCounter_;
  *out << "</td>" << endl; 
  *out << "</tr>"							<< endl; 


  *out << "</table>"							<< endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;

  *out << "<tr>"                                             << endl;
  *out << "<th colspan=2>"                                   << endl;
  *out << "<textarea rows=" << 5 << " cols=50 scroll=yes";
  *out << " readonly title=\"Last Log Messages\">"		     << endl;
  *out << logsAsString()                                         << endl;
  *out << "</textarea></td></tr></table>"                        << endl;

  // lumisection summary table
  *out << "   <table border=1 bgcolor=\"#CFCFCF\">" << endl;
  *out << "     <tr>"							<< endl;
  *out << "       <td> LS </td>";
  if(rollingLsWrap_)
    {
      for(unsigned int i = rollingLsIndex_; i < lumiSectionsCtr_.size(); i++)
	*out << "<td " << (lumiSectionsTo_[i] ? "bgcolor=\"red\"" : "")
	     << ">" << lumiSectionsCtr_[i].first << "</td>" << endl;
      for(unsigned int i = 0; i < rollingLsIndex_; i++)
	*out << "<td " << (lumiSectionsTo_[i] ? "bgcolor=\"red\"" : "")
	     << ">" << lumiSectionsCtr_[i].first << "</td>" << endl;
    }
  else
      for(unsigned int i = rollingLsIndex_; i < lumiSectionsCtr_.size(); i++)
	*out << "<td  " << (lumiSectionsTo_[i] ? "bgcolor=\"red\"" : "")
	     << ">" << lumiSectionsCtr_[i].first << "</td>" << endl;

  *out << "     </tr>"							<< endl;    
  *out << "     <tr>"							<< endl;
  *out << "       <td> Ev </td>";
  if(rollingLsWrap_)
    {
      for(unsigned int i = rollingLsIndex_; i < lumiSectionsCtr_.size(); i++)
	*out << "<td>" << lumiSectionsCtr_[i].second << "</td>" << endl;
      for(unsigned int i = 0; i < rollingLsIndex_; i++)
	*out << "<td>" << lumiSectionsCtr_[i].second << "</td>" << endl;
    }
  else
      for(unsigned int i = rollingLsIndex_; i < lumiSectionsCtr_.size(); i++)
	*out << "<td>" << lumiSectionsCtr_[i].second << "</td>" << endl;
  *out << "     </tr>"							<< endl;    
  *out << "</table>" << endl;

  ServiceWebRegistry *swr = 0;
  if(0!=evtProcessor_)
    {
      edm::ServiceRegistry::Operate operate(evtProcessor_->getToken());
      try{
	if(edm::Service<ServiceWebRegistry>().isAvailable())
	  swr = edm::Service<ServiceWebRegistry>().operator->();
      }
      catch(...) {
	LOG4CPLUS_WARN(getApplicationLogger(),
		       "Exception when trying to get service ServiceWebRegistry");
      }
    }
  if(0!=swr)
    {
      std::vector<ServiceWeb *> swebs = swr->getWebs();
      *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">" << endl;
      //      *out << "<colgroup> <colgroup align=\"right\">"                    << endl;
      *out << "  <tr>"                                                   << endl;
      *out << "    <th colspan=2>"                                       << endl;
      *out << "      " << "Linked Services"                              << endl;
      *out << "    </th>"                                                << endl;
      *out << "  </tr>"                                                  << endl;
      
      *out << "<tr>" << endl;
      *out << "<th >" << endl;
      *out << "Service" << endl;
      *out << "</th>" << endl;
      *out << "<th>" << endl;
      *out << "Address" << endl;
      *out << "</th>" << endl;
      *out << "</tr>" << endl;
      for(unsigned int i = 0; i < swebs.size(); i++)
	{
	  *out << " <tr><td><a href=\"/" << urn << "/serviceWeb?service=" 
	       << swebs[i]->name() << "\">" << swebs[i]->name() 
	       << "</a></td><td>" << hex << (unsigned int)swebs[i] 
	       << dec << "</td></tr>" << endl;
	}
      *out << "</table>" << endl;
    }
  *out << "</table>"							<< endl;
  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;

}


//______________________________________________________________________________
void FUEventProcessor::taskWebPage(xgi::Input *in, xgi::Output *out,const string &urn)
{
  ModuleWebRegistry *mwr = 0;
  edm::ServiceRegistry::Operate operate(evtProcessor_->getToken());
  try{
    if(edm::Service<ModuleWebRegistry>().isAvailable())
      mwr = edm::Service<ModuleWebRegistry>().operator->();
  }
  catch(...) {
    LOG4CPLUS_WARN(getApplicationLogger(),
		   "Exception when trying to get service ModuleWebRegistry");
  }
  TimeProfilerService *tpr = 0;
  try{
    if(edm::Service<TimeProfilerService>().isAvailable())
      tpr = edm::Service<TimeProfilerService>().operator->();
  }
  catch(...) { 
  }

  *out << "<td>" << endl;
  //Process details table
  *out << "<table frame=\"void\" rules=\"rows\" class=\"modules\">"	<< endl;
  *out << "  <tr>"							<< endl;
  *out << "    <th colspan=3>"						<< endl;
  *out << "      " << "HLT"						<< endl;
  if(descs_.size()>0)
    *out << " (Process " << descs_[0]->processName() << ")"		<< endl;
  *out << "    </th>"							<< endl;
  *out << "  </tr>"							<< endl;

  *out << "  <tr >"							<< endl;
  *out << "    <th >"							<< endl;
  *out << "       Module"						<< endl;
  *out << "    </th>"							<< endl;
  *out << "    <th >"							<< endl;
  *out << "       Label"						<< endl;
  *out << "    </th>"							<< endl;
  *out << "    <th >"							<< endl;
  *out << "       Version"						<< endl;
  *out << "    </th>"							<< endl;
  if(tpr)
    {
      *out << "    <th >"                                                       << endl;
      *out << "       first"                                            << endl;
      *out << "    </th>"                                                       << endl;
      *out << "    <th >"                                                       << endl;
      *out << "       ave"                                              << endl;
      *out << "    </th>"                                                       << endl;
      *out << "    <th >"                                                       << endl;
      *out << "       max"                                              << endl;
      *out << "    </th>"                                                       << endl;
    }
  *out << "  </tr>"							<< endl;
  
  for(unsigned int idesc = 0; idesc < descs_.size(); idesc++)
    {
      *out << "  <tr>"							<< endl;
      *out << "    <td >";
      if(mwr && mwr->checkWeb(descs_[idesc]->moduleName()))
	*out << "<a href=\"/" << urn 
	     << "/moduleWeb?module=" 
	     << descs_[idesc]->moduleName() << "\">" 
	     << descs_[idesc]->moduleName() << "</a>";
      else
	*out << descs_[idesc]->moduleName();
      *out << "</td>"							<< endl;
      *out << "    <td >";
      *out << descs_[idesc]->moduleLabel();
      *out << "</td>"							<< endl;
      *out << "    <td >";
      *out << descs_[idesc]->releaseVersion();
      *out << "</td>"							<< endl;
      if(tpr)
        {
          *out << "    <td align=\"right\">";
          *out << tpr->getFirst(descs_[idesc]->moduleLabel());
          *out << "</td>"                                                       << endl;
          *out << "    <td align=\"right\"";
	  *out << (tpr->getAve(descs_[idesc]->moduleLabel())>1. ? "bgcolor=\"red\"" : "") 
	       << ">";
          *out << tpr->getAve(descs_[idesc]->moduleLabel());
          *out << "</td>"                                                       << endl;
          *out << "    <td align=\"right\">";
          *out << tpr->getMax(descs_[idesc]->moduleLabel());
          *out << "</td>"                                                       << endl;
        }
      *out << "  </tr>" << endl;
    }
  *out << "</table>" << endl;
  *out << "</td>" << endl;
}


//______________________________________________________________________________
void FUEventProcessor::moduleWeb(xgi::Input  *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  Cgicc cgi(in);
  vector<FormEntry> el1;
  cgi.getElement("module",el1);
  if(evtProcessor_)  {
    if(el1.size()!=0) {
      string mod = el1[0].getValue();
      edm::ServiceRegistry::Operate operate(evtProcessor_->getToken());
      ModuleWebRegistry *mwr = 0;
      try{
	if(edm::Service<ModuleWebRegistry>().isAvailable())
	  mwr = edm::Service<ModuleWebRegistry>().operator->();
      }
      catch(...) { 
	LOG4CPLUS_WARN(getApplicationLogger(),
		       "Exception when trying to get service ModuleWebRegistry");
      }
      mwr->invoke(in,out,mod);
    }
  }
  else {
    *out<<"EventProcessor just disappeared "<<endl;
  }
}


//______________________________________________________________________________
void FUEventProcessor::serviceWeb(xgi::Input  *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  Cgicc cgi(in);
  vector<FormEntry> el1;
  cgi.getElement("service",el1);
  if(evtProcessor_)  {
    if(el1.size()!=0) {
      string ser = el1[0].getValue();
      edm::ServiceRegistry::Operate operate(evtProcessor_->getToken());
      ServiceWebRegistry *swr = 0;
      try{
	if(edm::Service<ServiceWebRegistry>().isAvailable())
	  swr = edm::Service<ServiceWebRegistry>().operator->();
      }
      catch(...) { 
	LOG4CPLUS_WARN(getApplicationLogger(),
		       "Exception when trying to get service ModuleWebRegistry");
      }
      swr->invoke(in,out,ser);
    }
  }
  else {
    *out<<"EventProcessor just disappeared "<<endl;
  }
}


//______________________________________________________________________________


void FUEventProcessor::spotlightWebPage(xgi::Input  *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  xdata::InfoSpace *ispace = getApplicationInfoSpace();
  string urn = getApplicationDescriptor()->getURN();
  ostringstream ourl;
  ourl << "'/" <<  urn << "/microState'";
  *out << "<!-- base href=\"/" <<  urn
       << "\"> -->" << endl;
  *out << "<html>"                                                   << endl;
  *out << "<head>"                                                   << endl;
  *out << "<link type=\"text/css\" rel=\"stylesheet\"";
  *out << " href=\"/" <<  urn
       << "/styles.css\"/>"                   << endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName() 
       << getApplicationDescriptor()->getInstance() 
       << " MAIN</title>"     << endl;
  *out << "</head>"                                                  << endl;
  *out << "<body onload=\"loadXMLDoc()\">"                           << endl;
  *out << "<table border=\"0\" width=\"100%\">"                      << endl;
  *out << "<tr>"                                                     << endl;
  *out << "  <td align=\"left\">"                                    << endl;
  *out << "    <img"                                                 << endl;
  *out << "     align=\"middle\""                                    << endl;
  *out << "     src=\"/evf/images/spoticon.jpg\""			     << endl;
  *out << "     alt=\"main\""                                        << endl;
  *out << "     width=\"64\""                                        << endl;
  *out << "     height=\"64\""                                       << endl;
  *out << "     border=\"\"/>"                                       << endl;
  *out << "    <b>"                                                  << endl;
  *out << getApplicationDescriptor()->getClassName() 
       << getApplicationDescriptor()->getInstance()                  << endl;
  *out << "      " << fsm_.stateName()->toString()                   << endl;
  *out << "    </b>"                                                 << endl;
  *out << "  </td>"                                                  << endl;
  *out << "  <td width=\"32\">"                                      << endl;
  *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << endl;
  *out << "      <img"                                               << endl;
  *out << "       align=\"middle\""                                  << endl;
  *out << "       src=\"/hyperdaq/images/HyperDAQ.jpg\""             << endl;
  *out << "       alt=\"HyperDAQ\""                                  << endl;
  *out << "       width=\"32\""                                      << endl;
  *out << "       height=\"32\""                                     << endl;
  *out << "       border=\"\"/>"                                     << endl;
  *out << "    </a>"                                                 << endl;
  *out << "  </td>"                                                  << endl;
  *out << "  <td width=\"32\">"                                      << endl;
  *out << "  </td>"                                                  << endl;
  *out << "  <td width=\"32\">"                                      << endl;
  *out << "    <a href=\"/" << urn << "/\">"                         << endl;
  *out << "      <img"                                               << endl;
  *out << "       align=\"middle\""                                  << endl;
  *out << "       src=\"/evf/images/epicon.jpg\""		     << endl;
  *out << "       alt=\"main\""                                      << endl;
  *out << "       width=\"32\""                                      << endl;
  *out << "       height=\"32\""                                     << endl;
  *out << "       border=\"\"/>"                                     << endl;
  *out << "    </a>"                                                 << endl;
  *out << "  </td>"                                                  << endl;
  *out << "</tr>"                                                    << endl;
  *out << "</table>"                                                 << endl;

  *out << "<hr/>"                                                    << endl;
  *out << "<table>"                                                  << endl;

  *out << "<tr valign=\"top\">"                                      << endl;
  *out << "<td>" << endl;


  if(evtProcessor_ && !inRecovery_)
    {
      edm::TriggerReport tr; 
      evtProcessor_->getTriggerReport(tr);

      // trigger summary table
      *out << "<table border=1 bgcolor=\"#CFCFCF\">" << endl;
      *out << "  <tr>"							<< endl;
      *out << "    <th colspan=7>"						<< endl;
      *out << "      " << "Trigger Summary"					<< endl;
      *out << "    </th>"							<< endl;
      *out << "  </tr>"							<< endl;

      *out << "  <tr >"							<< endl;
      *out << "    <th >Path</th>"						<< endl;
      *out << "    <th >Exec</th>"						<< endl;
      *out << "    <th >Pass</th>"						<< endl;
      *out << "    <th >Fail</th>"						<< endl;
      *out << "    <th >Except</th>"					<< endl;
      *out << "    <th >TargetPF</th>"					<< endl;
      *out << "  </tr>"							<< endl;
      xdata::Serializable *psid = 0;
      try{
	psid = ispace->find("prescaleSetIndex");
      }
      catch(xdata::exception::Exception e){
      }


      for(unsigned int i=0; i<tr.trigPathSummaries.size(); i++) {
	*out << "  <tr>" << endl;
	*out << "    <td>"<< tr.trigPathSummaries[i].name << "</td>"		<< endl;
	*out << "    <td>" << tr.trigPathSummaries[i].timesRun << "</td>"		<< endl;
	*out << "    <td>" << tr.trigPathSummaries[i].timesPassed << "</td>"	<< endl;
	*out << "    <td >" << tr.trigPathSummaries[i].timesFailed << "</td>"	<< endl;
	*out << "    <td ";
	if(tr.trigPathSummaries[i].timesExcept !=0)
	  *out << "bgcolor=\"red\""		      					<< endl;
	*out << ">" << tr.trigPathSummaries[i].timesExcept << "</td>"		<< endl;
	if(psid != 0)
	  {
	    *out << "    <td>"
		 << prescaleSvc_->getPrescale(tr.trigPathSummaries[i].name) 
		 << "</td>"		<< endl;
	  }
	else 	*out << "    <td>N/A</td>"		                        << endl;
	*out << "  </tr >"								<< endl;
	
      }
    }
  else if(inRecovery_)
    {
      *out << "  <tr>"							<< endl;
      *out << "    <td bgcolor=\"red\"> In Recovery !!! </td>"	      		<< endl;
      *out << "  </tr>"							<< endl;
    }
  *out << "</table>" << endl;

  *out << "</td>" << endl;

  if(evtProcessor_)
    taskWebPage(in,out,urn);
  else
    *out << "<td>HLT Unconfigured</td>" << endl;
  *out << "</table>"                                                 << endl;
  
  *out << "<br><textarea rows=" << 10 << " cols=80 scroll=yes>"      << endl;
  *out << configuration_                                             << endl;
  *out << "</textarea><P>"                                           << endl;
  
  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;


}

//______________________________________________________________________________
void FUEventProcessor::microState(xgi::Input  *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  edm::ServiceRegistry::Operate operate(serviceToken_);
  MicroStateService *mss = 0;
  string micro1 = "unavailable";
  if(epInitialized_)
    micro1 = "initialized";
  string micro2 = "unavailable";
  if(0 != evtProcessor_ && evtProcessor_->getState() != edm::event_processor::sInit)
    {
      try{
	mss = edm::Service<MicroStateService>().operator->();
      }
      catch(...) { 
	LOG4CPLUS_INFO(getApplicationLogger(),
		       "exception when trying to get service MicroStateService");
      }
    }
  if(mss) {
    micro1 = evtProcessor_->currentStateName();
    micro2 = mss->getMicroState2();
  }
  *out << fsm_.stateName()->toString() << endl;   
  *out << "<br>  " << micro1 << endl;
  *out << "<br>  " << micro2 << endl;
  *out << "<br>  " << nbAccepted_.value_ << "/" << nbProcessed_.value_  
       << " (" << float(nbAccepted_.value_)/float(nbProcessed_.value_)*100. <<"%)" << endl;
  *out << "<br>  " << lsidAsString_ << "/" << lsidTimedOutAsString_ << endl;
  *out << "<br>  " << psidAsString_ << endl;
  *out << " " << endl;
}


void FUEventProcessor::attachDqmToShm() throw (evf::Exception)  
{
  string errmsg;
  bool success = false;
  try {
    edm::ServiceRegistry::Operate operate(evtProcessor_->getToken());
    if(edm::Service<FUShmDQMOutputService>().isAvailable())
      success = edm::Service<FUShmDQMOutputService>()->attachToShm();
    if (!success) errmsg = "Failed to attach DQM service to shared memory";
  }
  catch (cms::Exception& e) {
    errmsg = "Failed to attach DQM service to shared memory: " + (string)e.what();
  }
  if (!errmsg.empty()) XCEPT_RAISE(evf::Exception,errmsg);
}



void FUEventProcessor::detachDqmFromShm() throw (evf::Exception)
{
  string errmsg;
  bool success = false;
  try {
    edm::ServiceRegistry::Operate operate(evtProcessor_->getToken());
    if(edm::Service<FUShmDQMOutputService>().isAvailable())
      success = edm::Service<FUShmDQMOutputService>()->detachFromShm();
    if (!success) errmsg = "Failed to detach DQM service from shared memory";
  }
  catch (cms::Exception& e) {
    errmsg = "Failed to detach DQM service from shared memory: " + (string)e.what();
  }
  if (!errmsg.empty()) XCEPT_RAISE(evf::Exception,errmsg);
}


void FUEventProcessor::startScalersWorkLoop() throw (evf::Exception)
{
  
  try {
    wlScalers_=
      toolbox::task::getWorkLoopFactory()->getWorkLoop(sourceId_+"Scalers",
						       "waiting");
    if (!wlScalers_->isActive()) wlScalers_->activate();
    asScalers_ = toolbox::task::bind(this,&FUEventProcessor::scalers,
				      sourceId_+"Scalers");
    wlScalers_->submit(asScalers_);
    wlScalersActive_ = true;
  }
  catch (xcept::Exception& e) {
    string msg = "Failed to start workloop 'Scalers'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
}


void FUEventProcessor::startMonitoringWorkLoop() throw (evf::Exception)
{
  struct timezone timezone;
  gettimeofday(&monStartTime_,&timezone);
  
  try {
    wlMonitoring_=
      toolbox::task::getWorkLoopFactory()->getWorkLoop(sourceId_+"Monitoring",
						       "waiting");
    if (!wlMonitoring_->isActive()) wlMonitoring_->activate();
    asMonitoring_ = toolbox::task::bind(this,&FUEventProcessor::monitoring,
				      sourceId_+"Monitoring");
    wlMonitoring_->submit(asMonitoring_);
    wlMonitoringActive_ = true;
  }
  catch (xcept::Exception& e) {
    string msg = "Failed to start workloop 'Monitoring'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
}


//______________________________________________________________________________
bool FUEventProcessor::scalers(toolbox::task::WorkLoop* wl)
{
  ::sleep(1); //avoid synchronization issues at the start of the event loop
  edm::ServiceRegistry::Operate operate(serviceToken_);
  monitorInfoSpace_->lock();
  if(evtProcessor_)
    {
      edm::event_processor::State st = evtProcessor_->getState();
      monitorInfoSpace_->unlock();
      if(st == edm::event_processor::sRunning && fsm_.stateName()->toString()=="Enabled")
	{
	  if(!getTriggerReport(true)) {
	    wlScalersActive_ = false;
	    return false;
	  }
	  if(lastLsTimedOut_) lsidTimedOutAsString_ = localLsIncludingTimeOuts_.toString();
	  else lsidTimedOutAsString_ = "";
	  if(lastLsTimedOut_ && lastLsWithEvents_==lastLsWithTimeOut_) return true;
	  if(!fireScalersUpdate()){
	    wlScalersActive_ = false;
	    return false;
	  }
	}
      else 
	{
	  wlScalersActive_ = false;
	  return false;
	}
    }
  else
    {
      monitorInfoSpace_->unlock();
      wlScalersActive_ = false;
      return false;
    }
  squidPresent_ = squidnet_.check();
  return true;
}

//______________________________________________________________________________
bool FUEventProcessor::monitoring(toolbox::task::WorkLoop* wl)
{
  
  struct timeval  monEndTime;
  struct timezone timezone;
  gettimeofday(&monEndTime,&timezone);
  xdata::InfoSpace *ispace = getApplicationInfoSpace();
  edm::ServiceRegistry::Operate operate(serviceToken_);
  //detect failures of edm event processor and attempts recovery procedure
  if(evtProcessor_)
    {
      edm::event_processor::State st = evtProcessor_->getState();
      if(watching_ && fsm_.stateName()->toString()=="Enabled" && 
	 !(st == edm::event_processor::sRunning || st == edm::event_processor::sStopping))
	{
	  inRecovery_ = true;
	  LOG4CPLUS_WARN(getApplicationLogger(),
			 "failure detected in internal edm::EventProcessor - attempting local recovery procedure");
	  ModuleWebRegistry *mwr = 0;
	  try{
	    if(edm::Service<ModuleWebRegistry>().isAvailable())
	      mwr = edm::Service<ModuleWebRegistry>().operator->();
	  }
	  catch(...) { 
	    LOG4CPLUS_WARN(getApplicationLogger(),
			   "InRecovery::exception when trying to get service ModuleWebRegistry");
	  }
	  //update table for lumi section before going out of scope

	  triggerReportIncomplete_ = true;
	  edm::TriggerReport tr; 
	  evtProcessor_->getTriggerReport(tr);
	  unsigned int ls = 0;
	  try{
	    xdata::Serializable *lsid = ispace->find("lumiSectionIndex");
	    ls = ((xdata::UnsignedInteger32*)(lsid))->value_;
	  }
	  catch(xdata::exception::Exception e){
	  }
	  trh_.formatReportTable(tr,descs_);
	  trh_.triggerReportToTable(tr,ls);
	  //	  trh_.printReportTable();
	  if(mwr) 
	    {
	      mwr->clear();
	    }
	  int sc = 0;
	  if(hasShMem_) detachDqmFromShm();
	  //	  delete evtProcessor_;
	  epInitialized_ = false;
	  initEventProcessor();
	  evtProcessor_->beginJob();
	  if(hasShMem_) attachDqmToShm();
	  if(isRunNumberSetter_)
	    evtProcessor_->setRunNumber(runNumber_.value_);
	  else
	    evtProcessor_->declareRunNumber(runNumber_.value_);
	  try {
	    evtProcessor_->runAsync();
	    sc = evtProcessor_->statusAsync();
	  }
	  catch(cms::Exception &e) {
	    reasonForFailedState_ = e.explainSelf();
	    fsm_.fireFailed(reasonForFailedState_,this);
	    return false;
	  }    
	  catch(std::exception &e) {
	    reasonForFailedState_  = e.what();
	    fsm_.fireFailed(reasonForFailedState_,this);
	    return false;
	  }
	  catch(...) {
	    reasonForFailedState_ = "Unknown Exception";
	    fsm_.fireFailed(reasonForFailedState_,this);
	    return false;
	  }
	  
	  if(sc != 0) {
	    ostringstream oss;
	    oss<<"EventProcessor::runAsync returned status code " << sc;
	    reasonForFailedState_ = oss.str();
	    fsm_.fireFailed(reasonForFailedState_,this);
	    wlMonitoringActive_ = false;
	    return false;
	  }

	  //	  reasonForFailedState_ = "edm failure, EP state ";
	  //	  reasonForFailedState_ += evtProcessor_->currentStateName();
	  //	  fsm_.fireFailed(reasonForFailedState_,this);
	  LOG4CPLUS_WARN(getApplicationLogger(),
			 "edm::EventProcessor recovery completed successfully - please check operation of this and other nodes");
	  inRecovery_ = false;
	  recoveryCount_++;
	  localLog("-I- Recovery completed");
	  startScalersWorkLoop();
	}
    }

  MicroStateService *mss = 0;

  monitorInfoSpace_->lock();
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
	LOG4CPLUS_INFO(getApplicationLogger(),
		       "exception when trying to get service MicroStateService");
      }
      try{
	xdata::Serializable *lsid = ispace->find("lumiSectionIndex");
	if(lsid!=0){
	  lsidAsString_ = lsid->toString();
	}
      }
      catch(xdata::exception::Exception e){
	lsidAsString_ = "N/A";
      }
      xdata::Serializable *psid = 0;
      try{
	psid = ispace->find("prescaleSetIndex");
	if(psid!=0) {
	  psidAsString_ = psid->toString();
	}
      }
      catch(xdata::exception::Exception e){
	psidAsString_ = "N/A";
      }

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
  monitorInfoSpace_->unlock();  

  ::sleep(monSleepSec_.value_);
  
  return true;
}


bool FUEventProcessor::fireScalersUpdate()
{
  scalersUpdateAttempted_++;
  try{
    scalersInfoSpace_->lock();
    scalersInfoSpace_->fireItemGroupChanged(names_,0);
    scalersInfoSpace_->unlock();
  }
  catch(xdata::exception::Exception &e)
    {
      LOG4CPLUS_ERROR(getApplicationLogger(), "Exception from fireItemGroupChanged: " << e.what());
      localLog(e.what());
      return false;
    }
  typedef set<xdaq::ApplicationDescriptor*> AppDescSet_t;
  typedef AppDescSet_t::iterator            AppDescIter_t;
  
  AppDescSet_t rcms=
    getApplicationContext()->getDefaultZone()->
    getApplicationDescriptors("RCMSStateListener");
  if(rcms.size()==0) 
    {
	LOG4CPLUS_WARN(getApplicationLogger(),
		       "MonitorReceiver not found, perhaphs it has not been defined ? Scalers updater wl will bail out!");
	localLog("-W- MonitorReceiver not found, perhaphs it has not been defined ? Scalers updater wl will bail out!");
	return false;
    }
  AppDescIter_t it = rcms.begin();
  
  toolbox::net::URL url((*it)->getContextDescriptor()->getURL());
  toolbox::net::URL properurl(url.getProtocol(),url.getHost(),url.getPort(),"/rcms/servlet/monitorreceiver");
  xdaq::ContextDescriptor *ctxdsc = new xdaq::ContextDescriptor(properurl.toString());
  xdaq::ApplicationDescriptor *appdesc = new xdaq::ApplicationDescriptorImpl(ctxdsc,(*it)->getClassName(),(*it)->getLocalId(), "pippo");
  xdata::exdr::Serializer serializer;
  toolbox::net::URL at(getApplicationContext()->getContextDescriptor()->getURL() + "/" + getApplicationDescriptor()->getURN());
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

  try
    {
      serializer.exportAll( &scalersComplete_, &outBuffer );
    }
  catch(xdata::exception::Exception & e)
    {
      LOG4CPLUS_WARN(getApplicationLogger(),
		     "Exception in serialization of scalers table");      
      localLog("-W- Exception in serialization of scalers table");      
      return true;
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
    this->getApplicationContext()->postSOAP(msg,*(getApplicationDescriptor()),*appdesc);
  }
  catch(xdaq::exception::Exception &ex)
    {
      string message = "exception when posting SOAP message to MonitorReceiver";
      message += ex.what();
      LOG4CPLUS_WARN(getApplicationLogger(),message.c_str());
      string lmessage = "-W- "+message;
      localLog(lmessage);
      return true;
   }
  delete appdesc; 
  delete ctxdsc;
  scalersUpdateCounter_++;
  return true;
}

std::string FUEventProcessor::logsAsString()
{
  ostringstream oss;
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
  
void FUEventProcessor::localLog(string m)
{
  timeval tv;

  gettimeofday(&tv,0);
  tm *uptm = localtime(&tv.tv_sec);
  char datestring[256];
  strftime(datestring, sizeof(datestring),"%c", uptm);

  if(logRingIndex_ == 0){logWrap_ = true; logRingIndex_ = logRingSize_;}
  logRingIndex_--;
  ostringstream timestamp;
  timestamp << " at " << datestring;
  m += timestamp.str();
  logRing_[logRingIndex_] = m;
}

XDAQ_INSTANTIATOR_IMPL(evf::FUEventProcessor)
