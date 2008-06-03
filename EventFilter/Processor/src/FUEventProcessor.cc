////////////////////////////////////////////////////////////////////////////////
//
// FUEventProcessor
// ----------------
//
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/Processor/interface/FUEventProcessor.h"

#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/Utilities/interface/Exception.h"

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
  , isRunNumberSetter_(true)
  , outprev_(true)
  , monSleepSec_(1)
  , wlMonitoring_(0)
  , asMonitoring_(0)
  , reasonForFailedState_()
{
  //list of variables for scalers flashlist
  names_.push_back("lumiSectionIndex");
  names_.push_back("prescaleSetIndex");
  names_.push_back("scalersTable");


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
  ispace->fireItemAvailable("isRunNumberSetter",    &isRunNumberSetter_);
  ispace->fireItemAvailable("monSleepSec",          &monSleepSec_);

  ispace->fireItemAvailable("foundRcmsStateListener",fsm_.foundRcmsStateListener());
  
  
  ispace->fireItemAvailable("prescalerAsString",    &prescalerAsString_);
  //  ispace->fireItemAvailable("triggerReportAsString",&triggerReportAsString_);
  
  // Add infospace listeners for exporting data values
  getApplicationInfoSpace()->addItemChangedListener("parameterSet",        this);
  getApplicationInfoSpace()->addItemChangedListener("outputEnabled",       this);
  getApplicationInfoSpace()->addItemChangedListener("globalInputPrescale", this);
  getApplicationInfoSpace()->addItemChangedListener("globalOutputPrescale",this);

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

  std::stringstream oss3;
  oss3<<"urn:xdaq-scalers-"<<class_.toString();
  string monInfoSpaceName2=oss3.str();
  toolbox::net::URN urn2 = this->createQualifiedInfoSpace(monInfoSpaceName2);
  xdata::Table &stbl = trh_.getTable(); 
  scalersInfoSpace_ = xdata::getInfoSpaceFactory()->get(urn2.toString());
  scalersInfoSpace_->fireItemAvailable("scalersTable",      &stbl);
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
  
  //Get the trigger report.
  ModuleWebRegistry *mwr = 0;
  try{
    if(edm::Service<ModuleWebRegistry>().isAvailable())
      mwr = edm::Service<ModuleWebRegistry>().operator->();
  }
  catch(...) { 
    LOG4CPLUS_INFO(getApplicationLogger(),
		   "exception when trying to get service ModuleWebRegistry");
    
  }
  edm::TriggerReport tr; 
  if(mwr)
    {

      xdata::InfoSpace *ispace = getApplicationInfoSpace();
      unsigned int ls = 0;
      unsigned int ps = 0;
      xdata::Table::iterator it = scalersComplete_.begin();
      if( it == scalersComplete_.end())
	{
	  it = scalersComplete_.append();
	}
      if(useLock) {
	mwr->openBackDoor("DaqSource");
      }
      if(!inRecovery_)evtProcessor_->getTriggerReport(tr);
      try{
	xdata::Serializable *lsid = ispace->find("lumiSectionIndex");
	if(lsid) ls = ((xdata::UnsignedInteger32*)(lsid))->value_;
	xdata::Serializable *psid = ispace->find("prescaleSetIndex");
	if(psid) ps = ((xdata::UnsignedInteger32*)(psid))->value_;
	it->setField("lsid",*lsid);
	it->setField("psid",*psid);
      }
      catch(xdata::exception::Exception e){
      }

      if(useLock){
	mwr->closeBackDoor("DaqSource");
      }


      if(inRecovery_) { return false;}
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

      }
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
  
  return false;
}


//______________________________________________________________________________
bool FUEventProcessor::enabling(toolbox::task::WorkLoop* wl)
{
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
      return false;
    }
    
    LOG4CPLUS_INFO(getApplicationLogger(),"Finished enabling!");
    
    fsm_.fireEvent("EnableDone",this);
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "enabling FAILED: " + (string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
  }
  startScalersWorkLoop();
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
	fsm_.fireFailed(reasonForFailedState_,this);

      }
    if(hasShMem_) detachDqmFromShm();
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "stopping FAILED: " + (string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
  }
  
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
	fsm_.fireFailed(reasonForFailedState_,this);
      }
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "halting FAILED: " + (string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
  }
  
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
  
  cout << "before getting configuration " <<endl;
  // job configuration string
  ParameterSetRetriever pr(configString_.value_);
  configuration_ = pr.getAsString();
  cout << "after getting configuration " <<endl;
  
  boost::shared_ptr<edm::ParameterSet> params; // change this name!
  boost::shared_ptr<vector<edm::ParameterSet> > pServiceSets;
  try{
    makeParameterSets(configuration_, params, pServiceSets);
  }
  catch(cms::Exception &e){
    reasonForFailedState_ = e.explainSelf();
    fsm_.fireFailed(reasonForFailedState_,this);
    return;
  } 
  cout << "before making services " <<endl;
  // add default set of services
  if(!servicesDone_) {
    cout << "making services " <<endl;
    internal::addServiceMaybe(*pServiceSets,"DQMStore");
    //    internal::addServiceMaybe(*pServiceSets,"MonitorDaemon");
    internal::addServiceMaybe(*pServiceSets,"MLlog4cplus");
    internal::addServiceMaybe(*pServiceSets,"MicroStateService");
    if(hasPrescaleService_) internal::addServiceMaybe(*pServiceSets,"PrescaleService");
    
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
  cout << "after making services " <<endl;
  
  edm::ServiceRegistry::Operate operate(serviceToken_);


  //test rerouting of fwk logging to log4cplus
  edm::LogInfo("FUEventProcessor")<<"started MessageLogger Service.";
  edm::LogInfo("FUEventProcessor")<<"Using config string \n"<<configuration_;

  cout << "after logger info " <<endl;
  // instantiate the event processor
  try{
    vector<string> defaultServices;
    defaultServices.push_back("MessageLogger");
    defaultServices.push_back("InitRootHandlers");
    defaultServices.push_back("JobReportService");
    cout << "before deleting eventprocessor" << endl;
    monitorInfoSpace_->lock();
    if (0!=evtProcessor_) delete evtProcessor_;
    cout << "after deleting eventprocessor" << endl;
    cout << "before making eventprocessor " <<endl;
    evtProcessor_ = new edm::EventProcessor(configuration_,
					    serviceToken_,
					    edm::serviceregistry::kTokenOverrides,
					    defaultServices);
    cout << "after making eventprocessor " <<endl;
    monitorInfoSpace_->unlock();
    //    evtProcessor_->setRunNumber(runNumber_.value_);

    if(!outPut_)
      //evtProcessor_->toggleOutput();
      //evtProcessor_->prescaleInput(inputPrescale_);
      //evtProcessor_->prescaleOutput(outputPrescale_);
      evtProcessor_->enableEndPaths(outPut_);
    
    outprev_=outPut_;
    
    // to publish all module names to XDAQ infospace
    ModuleWebRegistry *mwr = 0;
    try{
      if(edm::Service<ModuleWebRegistry>().isAvailable())
	mwr = edm::Service<ModuleWebRegistry>().operator->();
    }
    catch(...) { 
      LOG4CPLUS_INFO(getApplicationLogger(),
		     "exception when trying to get service ModuleWebRegistry");
    }
    cout << "after publishing stuff " <<endl;
    if(mwr) 
      {
	mwr->publish(getApplicationInfoSpace());
      }
    if(mwr) 
      {
	mwr->publishToXmas(scalersInfoSpace_);
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
  macro_state_legend_ = oss.str();
  //fill microstate legend information
  descs_ = evtProcessor_->getAllModuleDescriptions();
  trh_.resetFormat();
  std::stringstream oss2;
  unsigned int outcount = 0;
  oss2 << 0 << "=In ";
  modmap_["IN"]=0;
  for(unsigned int j = 0; j < descs_.size(); j++)
    {
      if(descs_[j]->moduleName() == "ShmStreamConsumer") // find something better than hardcoding name
	{ 
	  outcount++;
	  oss2 << outcount << "=Out" << outcount << " ";
	  modmap_[descs_[j]->moduleLabel()]=outcount;
	  i++;
	}
    }
  unsigned int modcount = 0;
  for(i = 0; i < descs_.size(); i++)
    {
      if(descs_[i]->moduleName() != "ShmStreamConsumer")
	{
	  modcount++;
	  oss2 << outcount+modcount << "=" << descs_[i]->moduleLabel() << " ";
	  modmap_[descs_[i]->moduleLabel()]=outcount+modcount;
	}
    }
  micro_state_legend_ = oss2.str();
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
  cout << "getting infospace " << endl;
  xdata::InfoSpace *ispace = getApplicationInfoSpace();
  string urn = getApplicationDescriptor()->getURN();
  ostringstream ourl;
  ourl << "'/" <<  urn << "/microState'";
  *out << "<!-- base href=\"/" <<  urn
       << "\"> -->" << endl;
  *out << "<html>"                                                   << endl;
  *out << "<head>"                                                   << endl;
  //insert javascript code
  jsGen(in,out,ourl.str());
  *out << "<STYLE type=\"text/css\">"				     << endl;
  *out << "#T1 {"						     << endl;
  *out << "border-width: 2px; border: solid blue; text-align: left; ";
  *out << "background: lightgrey "				     << endl;
  *out << "}"							     << endl; 
  *out << "</STYLE> "						     << endl; 
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
  *out << "     src=\"/evf/images/epicon.jpg\""			     << endl;
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
  *out << "    <a href=\"/" << urn 
       << "/Spotlight\">"                                            << endl;
  *out << "      <img"                                               << endl;
  *out << "       align=\"middle\""                                  << endl;
  *out << "       src=\"/evf/images/spoticon.jpg\""		     << endl;
  *out << "       alt=\"debug\""                                     << endl;
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
  *out << "  <td>"                                                   << endl;
  *out << "    <table><tr><td>"                                      << endl;
  *out << "<div id=\"T1\" style=\"border:2px solid "		     << endl;
  if(fsm_.stateName()->value_ == "Failed")
    {
      *out << "red;height:80;width:150\">"			     << endl;
      *out << "<table><tr><td id=\"s1\">microState</td></tr>"	     << endl;
      *out << "</table>"					     << endl;
      *out << "</div><br /> "					     << endl;
      *out << "</td></tr><tr><td>"				     << endl;
      *out << "<textarea rows=" << 5 << " cols=50 scroll=yes";
      *out << " readonly title=\"Reason For Failed\">"		     << endl;
      *out << reasonForFailedState_                                  << endl;
      *out << "</textarea></td></tr></table>"                        << endl;
    }
  else
    {
      *out << "blue;height:80;width:150\">"			     << endl;
      *out << "<table><tr><td id=\"s1\">microState</td></tr>"	     << endl;
      *out << "</table>"					     << endl;
      *out << "</div><br /> "					     << endl;
      *out << "</td></tr></table>"				     << endl;
    }
  *out << "  </td>"						     << endl;
  *out << "  <td>"                                                   << endl;
  *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">" << endl;
  *out << "<colgroup> <colgroup align=\"rigth\">"                    << endl;
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
  /* obsolete 
  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Plugin Path" << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << getenv("SEAL_PLUGINS") << endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  */
  *out << "<tr>" << endl;
  *out << "<td >" << endl;
  *out << "Run Number" << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << runNumber_.toString() << endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  cout << "trying to get lsid " << endl;
  try{
    xdata::Serializable *lsid = ispace->find("lumiSectionIndex");
    *out << "<tr>" << endl;
    *out << "<td >" << endl;
    *out << "Luminosity Section" << endl;
    *out << "</td>" << endl;
    *out << "<td>" << endl;
    *out << lsid->toString() << endl;
    cout << "got lsid " << endl;
    *out << "</td>" << endl;
    *out << "</tr>"                                            << endl;
  }
  catch(xdata::exception::Exception e){
  }
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
  *out << "Monitor Sleep (s)" << endl;
  *out << "</td>" << endl;
  *out << "<td>" << endl;
  *out << monSleepSec_.toString() << endl;
  *out << "</td>" << endl;
  *out << "</tr>"                                            << endl;
  *out << "</table>" << endl;
  *out << "</tr>"                                            << endl;

  if(evtProcessor_)
    {
      edm::TriggerReport tr; 
      evtProcessor_->getTriggerReport(tr);

      *out << "<tr valign=\"top\">"						<< endl;
      *out << "<td>"							<< endl;
      //status table
      *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">"	<< endl;
      *out << "<colgroup> <colgroup align=\"rigth\">"			<< endl;
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
      *out << "  <tr>"							<< endl;
      *out << "    <td >"							<< endl;
      *out << "       EP state"						<< endl;
      *out << "    </td>"							<< endl;
      *out << "    <td>"							<< endl;
      *out << "      " << evtProcessor_->currentStateName()			<< endl;
      *out << "    </td>"							<< endl;
      *out << "  </tr>"							<< endl;
      *out << "  <tr>"							<< endl;
      *out << "    <td>"							<< endl;
      *out << "       edm::EP initialized"					<< endl;
      *out << "    </td>"							<< endl;
      *out << "    <td>"							<< endl;
      *out << "      " <<epInitialized_					<< endl;
      *out << "    </td>"							<< endl;
      *out << "  </tr>"							<< endl;
      *out << "  <tr>"							<< endl;
      *out << "    <td >"							<< endl;
      *out << "       Processed Events/Accepted Events"			<< endl;
      *out << "    </td>"							<< endl;
      *out << "    <td>"							<< endl;
      *out << "      " << evtProcessor_->totalEvents() << "/" 
	   << evtProcessor_->totalEventsPassed()				<< endl;
      *out << "    </td>"							<< endl;
      *out << "  </tr>"							<< endl;
      *out << "  <tr>"							<< endl;
      *out << "    <td>Endpaths State</td>"					<< endl;
      *out << "    <td";
      *out << (evtProcessor_->endPathsEnabled() ?  "> enabled" : 
	       " bgcolor=\"red\"> disabled" );
      *out << "    </td>"							<< endl;
      *out << "  </tr>"							<< endl;
      /* obsolete
       *out << "  <tr>"							<< endl;
       *out << "    <td >Global Input Prescale</td>"				<< endl;
       *out << "    <td> N/A this version</td>"				<< endl;
       *out << "  </tr>"							<< endl;
       *out << "  <tr>"							<< endl;
       *out << "    <td >Global Output Prescale</td>"			<< endl;
       *out << "    <td>N/A this version</td>"				<< endl;
       *out << "  </tr>"							<< endl;
       */
      *out << "</table>"							<< endl;

      *out << "<td>" << endl;
      // trigger summary table
      *out << "<table border=1 bgcolor=\"#CFCFCF\">" << endl;
      *out << "  <tr>"							<< endl;
      *out << "    <th colspan=5>"						<< endl;
      *out << "      " << "Trigger Summary"					<< endl;
      *out << "    </th>"							<< endl;
      *out << "  </tr>"							<< endl;

      *out << "  <tr >"							<< endl;
      *out << "    <th >Path</th>"						<< endl;
      *out << "    <th >Exec</th>"						<< endl;
      *out << "    <th >Pass</th>"						<< endl;
      *out << "    <th >Fail</th>"						<< endl;
      *out << "    <th >Except</th>"					<< endl;
      *out << "  </tr>"							<< endl;


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
	*out << "  </tr >"								<< endl;
	
      }
    }
  *out << "</table>" << endl;
  *out << "</td>" << endl;
  *out << "</tr>" << endl;
  
  
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
    cout<<"exception when trying to get the service registry "		<< endl;
  }
  TimeProfilerService *tpr = 0;
  try{
    if(edm::Service<TimeProfilerService>().isAvailable())
      tpr = edm::Service<TimeProfilerService>().operator->();
  }
  catch(...) { 
  }

  *out << "<tr colspan=2>" << endl;
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
          *out << "    <td align=\"right\">";
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
  *out << "</tr>" << endl;
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
	cout<<"exception when trying to get the service registry "<<endl;
      }
      mwr->invoke(in,out,mod);
    }
  }
  else {
    *out<<"EventProcessor just disappeared "<<endl;
  }
}


//______________________________________________________________________________
void FUEventProcessor::jsGen(xgi::Input *in, xgi::Output *out, string url)
  throw (xgi::exception::Exception)
{
  *out << "<script type=\"text/javascript\"> \n";
  *out << "var xmlhttp \n";
  *out << " \n";
  *out << "function loadXMLDoc() \n";
  *out << "{ \n";
  *out << "xmlhttp=null \n";
  *out << " \n";
  *out << "if (window.XMLHttpRequest) \n";
  *out << "  { \n";
  *out << "  xmlhttp=new XMLHttpRequest() \n";
  *out << "  } \n";
  *out << " \n";
  *out << "else if (window.ActiveXObject) \n";
  *out << "  { \n";
  *out << "  xmlhttp=new ActiveXObject(\"Microsoft.XMLHTTP\") \n";
  *out << "  } \n";
  *out << "if (xmlhttp!=null) \n";
  *out << "  { \n";
  *out << "  xmlhttp.onreadystatechange=state_Change \n";
  *out << "  xmlhttp.open(\"GET\"," << url << ",true) \n";
  *out << "  xmlhttp.send(null) \n";
  *out << "  setTimeout('loadXMLDoc()',500) \n";
  *out << "  } \n";
  *out << "else \n";
  *out << "  { \n";
  *out << "  alert(\"Your browser does not support XMLHTTP.\") \n";
  *out << "  } \n";
  *out << "} \n";
  *out << " \n";
  *out << "function state_Change() \n";
  *out << "{ \n";
  // if xmlhttp shows "loaded"
  *out << "if (xmlhttp.readyState==4) \n";
  *out << "  { \n";
  // if "OK" 
  *out << " if (xmlhttp.status==200) \n";
  *out << "  { \n";
  *out << "  document.getElementById('s1').innerHTML=xmlhttp.responseText \n";
  *out << "  } \n";
  *out << "  else \n";
  *out << "  { \n";
  *out << "  document.getElementById('s1').innerHTML=xmlhttp.statusText \n";
  *out << "  } \n";
  *out << "  } \n";
  *out << "} \n";
  *out << " \n";
  *out << "</script> \n";
}


void FUEventProcessor::spotlightWebPage(xgi::Input  *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
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
  *out << "    <a href=\"/" << urn 
       << "/Default\">"                                              << endl;
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
    micro1 = mss->getMicroState1();
    micro2 = mss->getMicroState2();
  }
  
  *out << micro1 << endl;
  *out << "<br>  " << micro2 << endl;
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
  }
  catch (xcept::Exception& e) {
    string msg = "Failed to start workloop 'Monitoring'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
}


//______________________________________________________________________________
bool FUEventProcessor::scalers(toolbox::task::WorkLoop* wl)
{
  edm::ServiceRegistry::Operate operate(serviceToken_);
  if(evtProcessor_)
    {
      ::sleep(1);
      edm::event_processor::State st = evtProcessor_->getState();
      if(st == edm::event_processor::sRunning)
	{
	  if(!getTriggerReport(true)) {return false;}
	  //	  trh_.printReportTable();
	  //	  scalersComplete_.writeTo(std::cout);
	  fireScalersUpdate();
	}
    }
  return true;
}

//______________________________________________________________________________
bool FUEventProcessor::monitoring(toolbox::task::WorkLoop* wl)
{
  
  struct timeval  monEndTime;
  struct timezone timezone;
  gettimeofday(&monEndTime,&timezone);
  edm::ServiceRegistry::Operate operate(serviceToken_);
  //detect failures of edm event processor and attempts recovery procedure
  if(evtProcessor_)
    {
      edm::event_processor::State st = evtProcessor_->getState();
      if(fsm_.stateName()->toString()=="Enabled" && 
	 !(st == edm::event_processor::sRunning || st == edm::event_processor::sStopping))
	{
	  inRecovery_ = true;
	  ModuleWebRegistry *mwr = 0;
	  try{
	    if(edm::Service<ModuleWebRegistry>().isAvailable())
	      mwr = edm::Service<ModuleWebRegistry>().operator->();
	  }
	  catch(...) { 
	    LOG4CPLUS_INFO(getApplicationLogger(),
			   "exception when trying to get service ModuleWebRegistry");
	  }
	  //update table for lumi section before going out of scope

	  triggerReportIncomplete_ = true;
	  edm::TriggerReport tr; 
	  evtProcessor_->getTriggerReport(tr);
	  xdata::InfoSpace *ispace = getApplicationInfoSpace();
	  unsigned int ls = 0;
	  try{
	    xdata::Serializable *lsid = ispace->find("lumiSectionIndex");
	    ls = ((xdata::UnsignedInteger32*)(lsid))->value_;
	  }
	  catch(xdata::exception::Exception e){
	  }
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
	  cout << "before initeventprocessor " << endl;
	  initEventProcessor();
	  cout << "after initeventprocessor " << endl;
	  evtProcessor_->beginJob();
	  cout << "after beginjob " << endl;
	  if(hasShMem_) attachDqmToShm();
	  cout << "after attachdqm " << endl;
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
	    return false;
	  }

	  //	  reasonForFailedState_ = "edm failure, EP state ";
	  //	  reasonForFailedState_ += evtProcessor_->currentStateName();
	  //	  fsm_.fireFailed(reasonForFailedState_,this);
	  inRecovery_ = false;
	  startScalersWorkLoop();
	}
    }

  MicroStateService *mss = 0;

  monitorInfoSpace_->lock();
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
  monitorInfoSpace_->unlock();

  ::sleep(monSleepSec_.value_);
  
  return true;
}


void FUEventProcessor::fireScalersUpdate()
{
  typedef set<xdaq::ApplicationDescriptor*> AppDescSet_t;
  typedef AppDescSet_t::iterator            AppDescIter_t;
  
  // locate input BU
  AppDescSet_t rcms=
    getApplicationContext()->getDefaultZone()->
    getApplicationDescriptors("MonitorReceiver");
  if(rcms.size()==0) std::cout << "Application MonitorReceiver not found" << std::endl;
  AppDescIter_t it = rcms.begin();

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
      std::cout << " BOH ? " << std::endl;
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
  //  msg->writeTo(std::cout);
  try{
    this->getApplicationContext()->postSOAP(msg,*(getApplicationDescriptor()),*(*it));
  }
  catch(xdaq::exception::Exception &ex)
    {
      std::cout << "Exception caught, message " << ex.what() << std::endl;
    }
}

XDAQ_INSTANTIATOR_IMPL(evf::FUEventProcessor)
