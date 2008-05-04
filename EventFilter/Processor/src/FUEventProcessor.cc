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

  dqmCollectorAddr_       = "localhost";
  dqmCollectorPort_       = 9090;
  dqmCollectorDelay_      = 5000;
  dqmCollectorReconDelay_ = 5;
  dqmCollectorSourceName_ = ns.str();

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
  
  ispace->fireItemAvailable("collectorAddr",        &dqmCollectorAddr_);
  ispace->fireItemAvailable("collectorPort",        &dqmCollectorPort_);
  ispace->fireItemAvailable("collSendUs",           &dqmCollectorDelay_);
  ispace->fireItemAvailable("collReconnSec",        &dqmCollectorReconDelay_);
  ispace->fireItemAvailable("monSourceName",        &dqmCollectorSourceName_);
  
  ispace->fireItemAvailable("prescalerAsString",    &prescalerAsString_);
  ispace->fireItemAvailable("triggerReportAsString",&triggerReportAsString_);
  
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
  
  // bind prescale related soap callbacks
  xoap::bind(this,&FUEventProcessor::getPsReport ,"GetPsReport",XDAQ_NS_URI);
  xoap::bind(this,&FUEventProcessor::getLsReport ,"GetLsReport",XDAQ_NS_URI);
  xoap::bind(this,&FUEventProcessor::putPrescaler,"PutPrescaler",XDAQ_NS_URI);
  
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
void FUEventProcessor::getTriggerReport(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  // Calling this method results in calling 
  // evtProcessor_->getTriggerReport, the value returned is encoded as
  // a string. This value is used to set the exported SOAP param :
  // 'triggerReportAsString_'. The FM then picks up this value use getParam...
  LOG4CPLUS_DEBUG(getApplicationLogger(),"getTriggerReport action invoked");
  
  //Get the trigger report.
  edm::TriggerReport tr; 
  evtProcessor_->getTriggerReport(tr);
  
  triggerReportAsString_ = triggerReportToString(tr);
  
  //Print the trigger report message in debug format.
  printTriggerReport(tr);
}


//______________________________________________________________________________
xoap::MessageReference FUEventProcessor::getPsReport(xoap::MessageReference msg)
  throw (xoap::exception::Exception)
{
  // callback to return the trigger statistics as a string
  // cout <<"getPsReport from cout " <<endl;
  LOG4CPLUS_DEBUG(getApplicationLogger(),"getPsReport from log4");

  // print request
  //msg->writeTo(std::cout);
  
  //Get the trigger report.
  edm::TriggerReport tr; 
  monitorInfoSpace_->lock();
  if(evtProcessor_)
    evtProcessor_->getTriggerReport(tr);
  monitorInfoSpace_->unlock();  
  // xdata::String ReportAsString = triggerReportToString(tr);
  string s = triggerReportToString(tr);
  
  // reply message
  try {
      xoap::MessageReference reply = xoap::createMessage();
      xoap::SOAPEnvelope envelope = reply->getSOAPPart().getEnvelope();
      xoap::SOAPBody body = envelope.getBody();
      xoap::SOAPName responseName = envelope.createName("getPsReportResponse", "xdaq", "XDAQ_NS_URI");
      xoap::SOAPBodyElement responseElement = body.addBodyElement(responseName);
      xoap::SOAPName attributeName = envelope.createName("state", "xdaq", "XDAQ_NS_URI");
      xoap::SOAPElement keyElement = responseElement.addChildElement(attributeName);
      keyElement.addTextNode(s);
      xoap::SOAPName attributeName2 = envelope.createName("psstatus", "xdaq", "XDAQ_NS_URI");
      xoap::SOAPElement keyElement2 = responseElement.addChildElement(attributeName2);
      //if(prescaleSvc_ != 0) {
      //keyElement2.addTextNode(prescaleSvc_->getStatus());
      //} else {
      keyElement2.addTextNode("!prescaleSvc_");
      //}
      return reply;

  }
  catch (xcept::Exception &e) {
    XCEPT_RETHROW(xoap::exception::Exception,
		  "Failed to create getPsReport response message",e);
  }
}


//______________________________________________________________________________
xoap::MessageReference FUEventProcessor::getLsReport(xoap::MessageReference msg)
  throw (xoap::exception::Exception)
{
  // callback to return the trigger statistics as a string
  // LOG4CPLUS_DEBUG(this->getApplicationLogger(), "setPsUpdate from log4");
  // cout <<"setPsUpdate from cout " <<endl;
  // msg->writeTo(std::cout);
  
  // decode 
  xoap::SOAPPart part = msg->getSOAPPart();
  xoap::SOAPEnvelope env = part.getEnvelope();
  xoap::SOAPBody msgbody = env.getBody();
  DOMNode* node = msgbody.getDOMNode();
  
  string requestString = "-1";
  DOMNodeList* bodyList = node->getChildNodes();
  for (unsigned int i = 0; i < bodyList->getLength(); i++) {
    DOMNode* command = bodyList->item(i);
    if (command->getNodeType() == DOMNode::ELEMENT_NODE) {
      std::string commandName = xoap::XMLCh2String (command->getLocalName());
      if ( commandName == "GetLsReport" ) {
	if ( command->hasAttributes() ) {
	  DOMNamedNodeMap * map = command->getAttributes();
	  for (int l=0 ; l< (int)map->getLength() ; l++) {
	    // loop over attributes of node
	    DOMNode * anode = map->item(l);
	    string attributeName = XMLString::transcode(anode->getNodeName());
	    if (attributeName == "lsAsString")
	      requestString = xoap::XMLCh2String(anode->getNodeValue());
	  }
	}
      }
    }
  }
  
  // reply message
  try {
      xoap::MessageReference reply = xoap::createMessage();
      xoap::SOAPEnvelope envelope = reply->getSOAPPart().getEnvelope();
      xoap::SOAPBody body = envelope.getBody();
      xoap::SOAPName responseName = envelope.createName("getLsReportResponse", "xdaq", "XDAQ_NS_URI");
      xoap::SOAPBodyElement responseElement = body.addBodyElement(responseName);
      xoap::SOAPName attributeName = envelope.createName("LS1", "xdaq", "XDAQ_NS_URI");
      xoap::SOAPElement keyElement = responseElement.addChildElement(attributeName);
      //if(prescaleSvc_ != 0) {
      //keyElement.addTextNode(prescaleSvc_->getLs(requestString));
      //} else {
      keyElement.addTextNode("!prescaleSvc_");
      //}
      xoap::SOAPName attributeName2 = envelope.createName("psstatus", "xdaq", "XDAQ_NS_URI");
      xoap::SOAPElement keyElement2 = responseElement.addChildElement(attributeName2);
      //if(prescaleSvc_ != 0) {
      //keyElement2.addTextNode(prescaleSvc_->getStatus());
      //} else {
      keyElement2.addTextNode("!prescaleSvc_");
      //}
      xoap::SOAPName attributeName3 = envelope.createName("psdebug", "xdaq", "XDAQ_NS_URI");
      xoap::SOAPElement keyElement3 = responseElement.addChildElement(attributeName3);
      keyElement3.addTextNode(requestString);
      return reply;
    
  }
  catch (xcept::Exception &e) {
    XCEPT_RETHROW(xoap::exception::Exception,
		  "Failed to create getLsUpdate response message", e);
  }
}


//______________________________________________________________________________
xoap::MessageReference FUEventProcessor::putPrescaler(xoap::MessageReference msg)
  throw (xoap::exception::Exception)
{
  //The EPSM has an exported SOAP param 'nextPrescalerAsString_' this is always
  //set first from the FM with the new prescaler value encoded as a string.
  //Next this function is called to pick up the new string value and fill the 
  //appropriate prescaler structure for addition to the prescaler cache...
  
  //  LOG4CPLUS_INFO(getApplicationLogger(),"putPrescaler action invoked");
  
  //  msg->writeTo(std::cout);
  //  cout << endl;
  
  string prescalerAsString = "INITIAL_VALUE";
  
  // decode 
  xoap::SOAPPart part = msg->getSOAPPart();
  xoap::SOAPEnvelope env = part.getEnvelope();
  xoap::SOAPBody msgbody = env.getBody();
  DOMNode* node = msgbody.getDOMNode();
  
  DOMNodeList* bodyList = node->getChildNodes();
  for (unsigned int i = 0; i < bodyList->getLength(); i++) {
    DOMNode* command = bodyList->item(i);
    if (command->getNodeType() == DOMNode::ELEMENT_NODE) {
      std::string commandName = xoap::XMLCh2String (command->getLocalName());
      if ( commandName == "PutPrescaler" ) {
	if ( command->hasAttributes() ) {
	  DOMNamedNodeMap * map = command->getAttributes();
	  for (int l=0 ; l< (int)map->getLength() ; l++) {
	    // loop over attributes of node
	    DOMNode * anode = map->item(l);
	    string attributeName = XMLString::transcode(anode->getNodeName());
	    if (attributeName == "prescalerAsString")
	      prescalerAsString =  xoap::XMLCh2String(anode->getNodeValue());
	  }
	}
      }
    }
  }
  
  //Get the prescaler string value. (Which was set by the FM)
  //  LOG4CPLUS_INFO(getApplicationLogger(),
  //		 "Using new prescaler string setting: "<<prescalerAsString);


  if ( prescalerAsString == "INITIAL_VALUE" ) {
    // cout << "prescalerAsString not updated, is " << prescalerAsString << endl;
  }
  else {
    //if(prescaleSvc_ != 0) {
    //The number of LS# to prescale module set associations in the prescale
    //cache.
    //int storeSize = prescaleSvc_->putPrescale(prescalerAsString);
    //LOG4CPLUS_DEBUG(getApplicationLogger(),
    //	      "prescaleSvc_->putPrescale(s): " << storeSize);
    //}
    //else {
    //LOG4CPLUS_DEBUG(getApplicationLogger(),"PrescaleService pointer == 0"); 
    //}
  }
  
  xoap::MessageReference reply = xoap::createMessage();
  xoap::SOAPEnvelope envelope = reply->getSOAPPart().getEnvelope();
  xoap::SOAPBody body = envelope.getBody();
  xoap::SOAPName responseName = envelope.createName("PutPrescalerResponse", "xdaq", "XDAQ_NS_URI");
  xoap::SOAPBodyElement responseElement = body.addBodyElement(responseName);
  xoap::SOAPName attributeName = envelope.createName("prescalerAsString", "xdaq", "XDAQ_NS_URI");
  xoap::SOAPElement keyElement = responseElement.addChildElement(attributeName);
  keyElement.addTextNode(prescalerAsString);
  xoap::SOAPName attributeName2 = envelope.createName("psstatus", "xdaq", "XDAQ_NS_URI");
  xoap::SOAPElement keyElement2 = responseElement.addChildElement(attributeName2);
  //if(prescaleSvc_ != 0) {
  //keyElement2.addTextNode(prescaleSvc_->getStatus());
  //} else {
  keyElement2.addTextNode("!prescaleSvc_");
  //}
  
  
  return reply;
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
  
  
  // job configuration string
  ParameterSetRetriever pr(configString_.value_);
  configuration_ = pr.getAsString();
  
  
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
  // add default set of services
  if(!servicesDone_) {
    //    internal::addServiceMaybe(*pServiceSets,"DaqMonitorROOTBackEnd");
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

  
  edm::ServiceRegistry::Operate operate(serviceToken_);

  /* this part is obsolete
  try{
    edm::Service<MonitorDaemon>()->rmt(dqmCollectorAddr_,
				       dqmCollectorPort_,
				       dqmCollectorDelay_,
				       dqmCollectorSourceName_,
				       dqmCollectorReconDelay_);
  }
  catch(...) { 
    LOG4CPLUS_DEBUG(getApplicationLogger(),
		   "exception when trying to get service MonitorDaemon");
  }
  */

  //test rerouting of fwk logging to log4cplus
  edm::LogInfo("FUEventProcessor")<<"started MessageLogger Service.";
  edm::LogInfo("FUEventProcessor")<<"Using config string \n"<<configuration_;


  // instantiate the event processor
  try{
    vector<string> defaultServices;
    defaultServices.push_back("MessageLogger");
    defaultServices.push_back("InitRootHandlers");
    defaultServices.push_back("JobReportService");
    
    monitorInfoSpace_->lock();
    if (0!=evtProcessor_) delete evtProcessor_;
    
    evtProcessor_ = new edm::EventProcessor(configuration_,
					    serviceToken_,
					    edm::serviceregistry::kTokenOverrides,
					    defaultServices);
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

    if(mwr) 
      {
	mwr->publish(getApplicationInfoSpace());
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
string FUEventProcessor::triggerReportToString(const edm::TriggerReport& tr)
{
  // Add an array length indicator so that the resulting string will have a 
  // little more readability.
  string ARRAY_LEN = "_";
  string SEPARATOR = " ";
  
  ostringstream oss;
  
  //TriggerReport::eventSummary
  oss<<tr.eventSummary.totalEvents<<SEPARATOR 
     <<tr.eventSummary.totalEventsPassed<<SEPARATOR
     <<tr.eventSummary.totalEventsFailed<<SEPARATOR;
  
  //TriggerReport::trigPathSummaries
  oss<<ARRAY_LEN<<tr.trigPathSummaries.size()<<SEPARATOR;
  for(unsigned int i=0; i<tr.trigPathSummaries.size(); i++) {
    oss<<tr.trigPathSummaries[i].bitPosition<<SEPARATOR 
       <<tr.trigPathSummaries[i].timesRun<<SEPARATOR
       <<tr.trigPathSummaries[i].timesPassed<<SEPARATOR
       <<tr.trigPathSummaries[i].timesFailed<<SEPARATOR
       <<tr.trigPathSummaries[i].timesExcept<<SEPARATOR
       <<tr.trigPathSummaries[i].name<<SEPARATOR;
    
    //TriggerReport::trigPathSummaries::moduleInPathSummaries
    oss<<ARRAY_LEN<<tr.trigPathSummaries[i].moduleInPathSummaries.size()<<SEPARATOR;
    for(unsigned int j=0;j<tr.trigPathSummaries[i].moduleInPathSummaries.size();j++) {
      oss<<tr.trigPathSummaries[i].moduleInPathSummaries[j].timesVisited<<SEPARATOR
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].timesPassed <<SEPARATOR
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].timesFailed <<SEPARATOR
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].timesExcept <<SEPARATOR
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].moduleLabel <<SEPARATOR;
    }
  }
  
  //TriggerReport::endPathSummaries
  oss<<ARRAY_LEN<<tr.endPathSummaries.size()<<SEPARATOR;
  for(unsigned int i=0; i<tr.endPathSummaries.size(); i++) {
    oss<<tr.endPathSummaries[i].bitPosition<<SEPARATOR 
       <<tr.endPathSummaries[i].timesRun<<SEPARATOR
       <<tr.endPathSummaries[i].timesPassed<<SEPARATOR
       <<tr.endPathSummaries[i].timesFailed<<SEPARATOR
       <<tr.endPathSummaries[i].timesExcept<<SEPARATOR
       <<tr.endPathSummaries[i].name<<SEPARATOR;
    
    //TriggerReport::endPathSummaries::moduleInPathSummaries
    oss<<ARRAY_LEN<<tr.endPathSummaries[i].moduleInPathSummaries.size()<<SEPARATOR;
    for(unsigned int j=0;j<tr.endPathSummaries[i].moduleInPathSummaries.size();j++) {
      oss<<tr.endPathSummaries[i].moduleInPathSummaries[j].timesVisited<<SEPARATOR
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].timesPassed <<SEPARATOR
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].timesFailed <<SEPARATOR
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].timesExcept <<SEPARATOR
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].moduleLabel <<SEPARATOR;
    }
  }
  
  //TriggerReport::workerSummaries
  oss<<ARRAY_LEN<<tr.workerSummaries.size()<<SEPARATOR;
  for(unsigned int i=0; i<tr.workerSummaries.size(); i++) {
    oss<<tr.workerSummaries[i].timesVisited<<SEPARATOR 
       <<tr.workerSummaries[i].timesRun    <<SEPARATOR
       <<tr.workerSummaries[i].timesPassed <<SEPARATOR
       <<tr.workerSummaries[i].timesFailed <<SEPARATOR
       <<tr.workerSummaries[i].timesExcept <<SEPARATOR
       <<tr.workerSummaries[i].moduleLabel <<SEPARATOR;
  }
  
  return oss.str();
}


//______________________________________________________________________________
void FUEventProcessor::printTriggerReport(const edm::TriggerReport& tr)
{
  ostringstream oss;
  
  oss<<"================================="<<"\n";
  oss<<"== BEGINNING OF TRIGGER REPORT =="<<"\n";
  oss<<"================================="<<"\n";
  oss<<"tr.eventSummary.totalEvents= "<<tr.eventSummary.totalEvents<<"\n" 
     <<"tr.eventSummary.totalEventsPassed= "<<tr.eventSummary.totalEventsPassed<<"\n"
     <<"tr.eventSummary.totalEventsFailed= "<<tr.eventSummary.totalEventsFailed<<"\n";
  
  oss<<"TriggerReport::trigPathSummaries"<<"\n";
  for(unsigned int i=0; i<tr.trigPathSummaries.size(); i++) {
    oss<<"tr.trigPathSummaries["<<i<<"].bitPosition = "
       <<tr.trigPathSummaries[i].bitPosition <<"\n" 
       <<"tr.trigPathSummaries["<<i<<"].timesRun = "
       <<tr.trigPathSummaries[i].timesRun <<"\n"
       <<"tr.trigPathSummaries["<<i<<"].timesPassed = "
       <<tr.trigPathSummaries[i].timesPassed <<"\n"
       <<"tr.trigPathSummaries["<<i<<"].timesFailed = "
       <<tr.trigPathSummaries[i].timesFailed <<"\n"
       <<"tr.trigPathSummaries["<<i<<"].timesExcept = "
       <<tr.trigPathSummaries[i].timesExcept <<"\n"
       <<"tr.trigPathSummaries["<<i<<"].name = "
       <<tr.trigPathSummaries[i].name <<"\n";
    
    //TriggerReport::trigPathSummaries::moduleInPathSummaries
    for(unsigned int j=0;j<tr.trigPathSummaries[i].moduleInPathSummaries.size();j++) {
      oss<<"tr.trigPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesVisited = "
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].timesVisited<<"\n"
	 <<"tr.trigPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesPassed = "
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].timesPassed<<"\n"
	 <<"tr.trigPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesFailed = "
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].timesFailed<<"\n"
	 <<"tr.trigPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesExcept = "
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].timesExcept<<"\n"
	 <<"tr.trigPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].moduleLabel = "
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].moduleLabel<<"\n";
    }
  }
  
  //TriggerReport::endPathSummaries
  for(unsigned int i=0;i<tr.endPathSummaries.size();i++) {
    oss<<"tr.endPathSummaries["<<i<<"].bitPosition = "
       <<tr.endPathSummaries[i].bitPosition <<"\n" 
       <<"tr.endPathSummaries["<<i<<"].timesRun = "
       <<tr.endPathSummaries[i].timesRun <<"\n"
       <<"tr.endPathSummaries["<<i<<"].timesPassed = "
       <<tr.endPathSummaries[i].timesPassed <<"\n"
       <<"tr.endPathSummaries["<<i<<"].timesFailed = "
       <<tr.endPathSummaries[i].timesFailed <<"\n"
       <<"tr.endPathSummaries["<<i<<"].timesExcept = "
       <<tr.endPathSummaries[i].timesExcept <<"\n"
       <<"tr.endPathSummaries["<<i<<"].name = "
       <<tr.endPathSummaries[i].name <<"\n";
    
    //TriggerReport::endPathSummaries::moduleInPathSummaries
    for(unsigned int j=0;j<tr.endPathSummaries[i].moduleInPathSummaries.size();j++) {
      oss<<"tr.endPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesVisited = "
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].timesVisited <<"\n"
	 <<"tr.endPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesPassed = "
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].timesPassed <<"\n"
	 <<"tr.endPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesFailed = "
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].timesFailed <<"\n"
	 <<"tr.endPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesExcept = "
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].timesExcept <<"\n"
	 <<"tr.endPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].moduleLabel = "
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].moduleLabel <<"\n";
    }
  }
  
  //TriggerReport::workerSummaries
  for(unsigned int i=0; i<tr.workerSummaries.size(); i++) {
    oss<<"tr.workerSummaries["<<i<<"].timesVisited = "
       <<tr.workerSummaries[i].timesVisited<<"\n" 
       <<"tr.workerSummaries["<<i<<"].timesRun = "
       <<tr.workerSummaries[i].timesRun<<"\n"
       <<"tr.workerSummaries["<<i<<"].timesPassed = "
       <<tr.workerSummaries[i].timesPassed <<"\n"
       <<"tr.workerSummaries["<<i<<"].timesFailed = "
       <<tr.workerSummaries[i].timesFailed <<"\n"
       <<"tr.workerSummaries["<<i<<"].timesExcept = "
       <<tr.workerSummaries[i].timesExcept <<"\n"
       <<"tr.workerSummaries["<<i<<"].moduleLabel = "
       <<tr.workerSummaries[i].moduleLabel <<"\n";
  }
  
  oss<<"==========================="<<"\n";
  oss<<"== END OF TRIGGER REPORT =="<<"\n";
  oss<<"==========================="<<"\n";
  
  LOG4CPLUS_DEBUG(getApplicationLogger(),oss.str());
}


//______________________________________________________________________________
void FUEventProcessor::defaultWebPage(xgi::Input  *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
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
bool FUEventProcessor::monitoring(toolbox::task::WorkLoop* wl)
{
  
  struct timeval  monEndTime;
  struct timezone timezone;
  gettimeofday(&monEndTime,&timezone);

  //detect failures of edm event processor and fire failed transition
  if(evtProcessor_)
    {
      edm::event_processor::State st = evtProcessor_->getState();
      if(fsm_.stateName()->toString()=="Enabled" && 
	 !(st == edm::event_processor::sRunning || st == edm::event_processor::sStopping))
	{
	  reasonForFailedState_ = "edm failure, EP state ";
	  reasonForFailedState_ += evtProcessor_->currentStateName();
	  fsm_.fireFailed(reasonForFailedState_,this);
	}
    }
  edm::ServiceRegistry::Operate operate(serviceToken_);
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



XDAQ_INSTANTIATOR_IMPL(evf::FUEventProcessor)
