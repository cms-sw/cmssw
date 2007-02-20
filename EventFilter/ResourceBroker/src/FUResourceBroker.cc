///////////////////////////////////////////////////////////////////////////////
//
// FUResourceBroker
// ----------------
//
//            10/20/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ResourceBroker/interface/FUResourceBroker.h"
#include "EventFilter/ResourceBroker/interface/BUProxy.h"

#include "EventFilter/Utilities/interface/RunBase.h"
#include "EventFilter/Utilities/interface/Crc.h"

#include "i2o/include/i2o/Method.h"

#include "interface/shared/include/i2oXFunctionCodes.h"

#include "xcept/include/xcept/tools.h"

#include "toolbox/include/toolbox/mem/HeapAllocator.h"
#include "toolbox/include/toolbox/mem/Reference.h"
#include "toolbox/include/toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/include/toolbox/mem/exception/Exception.h"

#include "xoap/MessageReference.h"
#include "xoap/MessageFactory.h"
#include "xoap/include/xoap/SOAPEnvelope.h"
#include "xoap/include/xoap/SOAPBody.h"
#include "xoap/include/xoap/domutils.h"
#include "xoap/Method.h"

#include <iostream>
#include <sstream>


using namespace std;
using namespace evf;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUResourceBroker::FUResourceBroker(xdaq::ApplicationStub *s)
  : xdaq::Application(s)
  , workLoopConfiguring_(0)
  , workLoopEnabling_(0)
  , workLoopStopping_(0)
  , workLoopHalting_(0)
  , asConfiguring_(0)
  , asEnabling_(0)
  , asStopping_(0)
  , asHalting_(0)
  , lock_(BSem::FULL)
  , gui_(0)
  , log_(getApplicationLogger())
  , i2oPool_(0)
  , resourceTable_(0)
  , instance_(0)
  , runNumber_(0)
  , nbShmClients_(0)
  , nbMBTot_(0.0)
  , nbMBPerSec_(0.0)
  , nbMBPerSecMin_(0.0)
  , nbMBPerSecMax_(0.0)
  , nbMBPerSecAvg_(0.0)
  , nbEvents_(0)
  , nbEventsPerSec_(0)
  , nbEventsPerSecMin_(0)
  , nbEventsPerSecMax_(0)
  , nbEventsPerSecAvg_(0)
  , nbAllocatedEvents_(0)
  , nbPendingRequests_(0)
  , nbReceivedEvents_(0)
  , nbDiscardedEvents_(0)
  , nbProcessedEvents_(0)
  , nbLostEvents_(0)
  , nbDataErrors_(0)
  , nbCrcErrors_(0)
  , shmMode_(false)
  , eventBufferSize_(4194304) // 4MB
  //, doDumpFragments_(false)
  , doDropEvents_(false)
  , doFedIdCheck_(true)
  , doCrcCheck_(1)
  , buClassName_("BU")
  , buInstance_(0)
  , queueSize_(64)
  , nbAllocateSent_(0)
  , nbTakeReceived_(0)
  , nbMeasurements_(0)
  , nbEventsLast_(0)
{
  //
  // setup the finite state machine
  //

  // action signatures
  asConfiguring_ =
    toolbox::task::bind(this,&FUResourceBroker::configuring,"configuring");
  asEnabling_ =
    toolbox::task::bind(this,&FUResourceBroker::enabling,"enabling");
  asStopping_ =
    toolbox::task::bind(this,&FUResourceBroker::stopping,"stopping");
  asHalting_ =
    toolbox::task::bind(this,&FUResourceBroker::halting,"halting");
  
  // work loops
  workLoopConfiguring_ =
    toolbox::task::getWorkLoopFactory()->getWorkLoop("Configuring",
						     "waiting");
  workLoopEnabling_ =
    toolbox::task::getWorkLoopFactory()->getWorkLoop("Enabling",
						     "waiting");
  workLoopStopping_ =
    toolbox::task::getWorkLoopFactory()->getWorkLoop("Stopping",
						     "waiting");
  workLoopHalting_ =
    toolbox::task::getWorkLoopFactory()->getWorkLoop("Halting",
						     "waiting");
  

  // bind SOAP callbacks
  xoap::bind(this,&FUResourceBroker::fsmCallback,"Configure",XDAQ_NS_URI);
  xoap::bind(this,&FUResourceBroker::fsmCallback,"Enable",   XDAQ_NS_URI);
  xoap::bind(this,&FUResourceBroker::fsmCallback,"Stop",     XDAQ_NS_URI);
  xoap::bind(this,&FUResourceBroker::fsmCallback,"Halt",     XDAQ_NS_URI);
  xoap::bind(this,&FUResourceBroker::fsmCallback,"Suspend",  XDAQ_NS_URI);
  xoap::bind(this,&FUResourceBroker::fsmCallback,"Resume",   XDAQ_NS_URI);
  
  // define finite state machine, states&transitions
  fsm_.addState('h', "halting"    ,this,&FUResourceBroker::fsmStateChanged);
  fsm_.addState('H', "Halted"     ,this,&FUResourceBroker::fsmStateChanged);
  fsm_.addState('c', "configuring",this,&FUResourceBroker::fsmStateChanged);
  fsm_.addState('R', "Ready"      ,this,&FUResourceBroker::fsmStateChanged);
  fsm_.addState('e', "enabling"   ,this,&FUResourceBroker::fsmStateChanged);
  fsm_.addState('E', "Enabled"    ,this,&FUResourceBroker::fsmStateChanged);
  fsm_.addState('s', "stopping"   ,this,&FUResourceBroker::fsmStateChanged);
  fsm_.addState('S', "Suspended"  ,this,&FUResourceBroker::fsmStateChanged);
  
  fsm_.addStateTransition('H', 'c', "Configure");
  fsm_.addStateTransition('c', 'R', "ConfigureDone");
  fsm_.addStateTransition('R', 'e', "Enable");
  fsm_.addStateTransition('e', 'E', "EnableDone");
  fsm_.addStateTransition('E', 's', "Stop");
  fsm_.addStateTransition('s', 'R', "StopDone");
  fsm_.addStateTransition('E', 'h', "Halt");
  fsm_.addStateTransition('R', 'h', "Halt");
  fsm_.addStateTransition('S', 'h', "Halt");
  fsm_.addStateTransition('h', 'H', "HaltDone");
  fsm_.addStateTransition('E', 'S', "Suspend",this,&FUResourceBroker::suspend);
  fsm_.addStateTransition('S', 'E', "Resume" ,this,&FUResourceBroker::resume);
  
  fsm_.setInitialState('H');
  fsm_.reset();
  stateName_ = fsm_.getStateName(fsm_.getCurrentState());
  
  if (!workLoopConfiguring_->isActive()) workLoopConfiguring_->activate();
  if (!workLoopEnabling_->isActive())    workLoopEnabling_   ->activate();
  if (!workLoopStopping_->isActive())    workLoopStopping_   ->activate();
  if (!workLoopHalting_->isActive())     workLoopHalting_    ->activate();
  
  
  // set source id in evf::RunBase
  url_     =
    getApplicationDescriptor()->getContextDescriptor()->getURL()+"/"+
    getApplicationDescriptor()->getURN();
  class_   =getApplicationDescriptor()->getClassName();
  instance_=getApplicationDescriptor()->getInstance();
  
  sourceId_=class_.toString()+"_"+instance_.toString();
  RunBase::sourceId_=sourceId_;
  
  // i2o callback for FU_TAKE messages from builder unit
  i2o::bind(this,
	    &FUResourceBroker::I2O_FU_TAKE_Callback,
	    I2O_FU_TAKE,
	    XDAQ_ORGANIZATION_ID);
  
  // bind HyperDAQ web pages
  xgi::bind(this,&evf::FUResourceBroker::webPageRequest,"Default");
  gui_=new WebGUI(this,&fsm_);
  vector<toolbox::lang::Method*> methods=gui_->getMethods();
  vector<toolbox::lang::Method*>::iterator it;
  for (it=methods.begin();it!=methods.end();++it) {
    if ((*it)->type()=="cgi") {
      string name=static_cast<xgi::MethodSignature*>(*it)->name();
      xgi::bind(this,&evf::FUResourceBroker::webPageRequest,name);
    }
  }
  
  // allocate i2o memery pool
  string i2oPoolName=sourceId_+"_i2oPool";
  try {
    toolbox::mem::HeapAllocator *allocator=new toolbox::mem::HeapAllocator();
    toolbox::net::URN urn("toolbox-mem-pool",i2oPoolName);
    toolbox::mem::MemoryPoolFactory* poolFactory=
      toolbox::mem::getMemoryPoolFactory();
    i2oPool_=poolFactory->createPool(urn,allocator);
  }
  catch (toolbox::mem::exception::Exception& e) {
    string s="Failed to create pool: "+i2oPoolName;
    LOG4CPLUS_FATAL(log_,s);
    XCEPT_RETHROW(xcept::Exception,s,e);
  }
  
  // publish all parameters to app info space
  exportParameters();
}


//______________________________________________________________________________
FUResourceBroker::~FUResourceBroker()
{

}



////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
xoap::MessageReference FUResourceBroker::fsmCallback(xoap::MessageReference msg)
  throw (xoap::exception::Exception)
{
  xoap::SOAPPart     part    =msg->getSOAPPart();
  xoap::SOAPEnvelope env     =part.getEnvelope();
  xoap::SOAPBody     body    =env.getBody();
  DOMNode           *node    =body.getDOMNode();
  DOMNodeList       *bodyList=node->getChildNodes();
  DOMNode           *command =0;
  string             commandName;
  
  for (UInt_t i=0;i<bodyList->getLength();i++) {
    command = bodyList->item(i);
    if(command->getNodeType() == DOMNode::ELEMENT_NODE) {
      commandName = xoap::XMLCh2String(command->getLocalName());
      break;
    }
  }
  
  if (commandName.empty()) {
    XCEPT_RAISE(xoap::exception::Exception,"Command not found.");
  }
  
  // fire appropriate event and create according response message
  try {
    toolbox::Event::Reference e(new toolbox::Event(commandName,this));
    fsm_.fireEvent(e);
    
    // response string
    xoap::MessageReference reply = xoap::createMessage();
    xoap::SOAPEnvelope envelope  = reply->getSOAPPart().getEnvelope();
    xoap::SOAPName responseName  = envelope.createName(commandName+"Response",
						       "xdaq",XDAQ_NS_URI);
    xoap::SOAPBodyElement responseElem =
      envelope.getBody().addBodyElement(responseName);
    
    // state string
    int               iState        = fsm_.getCurrentState();
    string            state         = fsm_.getStateName(iState);
    xoap::SOAPName    stateName     = envelope.createName("state",
							  "xdaq",XDAQ_NS_URI);
    xoap::SOAPElement stateElem     = responseElem.addChildElement(stateName);
    xoap::SOAPName    attributeName = envelope.createName("stateName",
							  "xdaq",XDAQ_NS_URI);
    stateElem.addAttribute(attributeName,state);
    
    return reply;
  }
  catch (toolbox::fsm::exception::Exception & e) {
    XCEPT_RETHROW(xoap::exception::Exception,"invalid command.",e);
  }	
}


//______________________________________________________________________________
void FUResourceBroker::fsmStateChanged(toolbox::fsm::FiniteStateMachine & fsm) 
  throw (toolbox::fsm::exception::Exception)
{
  stateName_   = fsm_.getStateName(fsm_.getCurrentState());
  string state = stateName_.toString();
  
  LOG4CPLUS_INFO(log_,"New state is: "<<state);
  
  if (state=="configuring") {
    try {
      workLoopConfiguring_->submit(asConfiguring_);
    }
    catch (xdaq::exception::Exception& e) {
      LOG4CPLUS_ERROR(log_,xcept::stdformat_exception_history(e));
    }
  }
  else if (state=="enabling") {
    try {
      workLoopEnabling_->submit(asEnabling_);
    }
    catch (xdaq::exception::Exception& e) {
      LOG4CPLUS_ERROR(log_,xcept::stdformat_exception_history(e));
    }
  }
  else if (state=="stopping") {
    try {
      workLoopStopping_->submit(asStopping_);
    }
    catch (xdaq::exception::Exception& e) {
      LOG4CPLUS_ERROR(log_,xcept::stdformat_exception_history(e));
    }
  }
  else if (state=="halting") {
    try {
      workLoopHalting_->submit(asHalting_);
    }
    catch (xdaq::exception::Exception& e) {
      LOG4CPLUS_ERROR(log_,xcept::stdformat_exception_history(e));
    }
  }
  
}

//______________________________________________________________________________
void FUResourceBroker::suspend(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  lock_.take();
  LOG4CPLUS_INFO(log_,"FUResourceBroker -> SUSPENDED <-");
}


//______________________________________________________________________________
void FUResourceBroker::resume(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  lock_.give();
  LOG4CPLUS_INFO(log_,"FUResourceBroker -> RESUMED <-");
}

   
//______________________________________________________________________________
bool FUResourceBroker::configuring(toolbox::task::WorkLoop* wl)
{
  LOG4CPLUS_INFO(log_, "Start configuring ...");
  connectToBUs();
  resourceTable_=new FUResourceTable(queueSize_,eventBufferSize_,shmMode_,
				     bu_[buInstance_],log_);
  resourceTable_->resetCounters();
  reset();
  LOG4CPLUS_INFO(log_, "Finished configuring!");
  
  toolbox::Event::Reference e(new toolbox::Event("ConfigureDone",this));
  fsm_.fireEvent(e);
  
  return false;
}


//______________________________________________________________________________
bool FUResourceBroker::enabling(toolbox::task::WorkLoop* wl)
{
  LOG4CPLUS_INFO(log_, "Start enabling ...");
  initTimer();
  if (shmMode_) resourceTable_->startWorkLoop();
  resourceTable_->sendAllocate();
  LOG4CPLUS_INFO(log_, "Finished enabling!");
  
  toolbox::Event::Reference e(new toolbox::Event("EnableDone",this));
  fsm_.fireEvent(e);
  
  return false;
}


//______________________________________________________________________________
bool FUResourceBroker::stopping(toolbox::task::WorkLoop* wl)
{
  LOG4CPLUS_INFO(log_, "Start stopping :) ...");
  while (resourceTable_->nbDiscarded()<resourceTable_->nbCompleted()) {
    LOG4CPLUS_INFO(log_,"Waiting for events to be consumed ...");
    ::sleep(1);
  }
  stopTimer();
  resourceTable_->shutDownClients();
  LOG4CPLUS_INFO(log_, "Finished stopping!");
  
  toolbox::Event::Reference e(new toolbox::Event("StopDone",this));
  fsm_.fireEvent(e);
  
  return false;
}


//______________________________________________________________________________
bool FUResourceBroker::halting(toolbox::task::WorkLoop* wl)
{
  LOG4CPLUS_INFO(log_, "Start halting ...");
  while (resourceTable_->nbDiscarded()<resourceTable_->nbCompleted()) {
    LOG4CPLUS_INFO(log_,"Waiting for events to be consumed ...");
    ::sleep(1);
  }
  stopTimer();
  resourceTable_->shutDownClients();
  UInt_t count = 0;
  while (count<10) {
    if (resourceTable_->nbShmClients()==0) {
      delete resourceTable_;
      resourceTable_=0;
      LOG4CPLUS_INFO(log_,++count<<". try to destroy resource table succeeded!");
      break;
    }
    else {
      LOG4CPLUS_WARN(log_,++count<<". try to destroy resource table failed ...");
      ::sleep(2);
    }
  } 
  if (0!=resourceTable_) LOG4CPLUS_ERROR(log_,"Failed to destroy resource table.");
  LOG4CPLUS_INFO(log_, "Finished halting!");
  
  toolbox::Event::Reference e(new toolbox::Event("HaltDone",this));
  fsm_.fireEvent(e);
  
  return false;
}


//______________________________________________________________________________
void FUResourceBroker::timeExpired(toolbox::task::TimerEvent& e)
{
  lock_.take();

  gui_->lockInfoSpaces();

  nbMeasurements_++;
 
  // number of events per second measurement
  nbEvents_      =resourceTable_->nbCompleted();
  nbEventsPerSec_=nbEvents_-nbEventsLast_;
  nbEventsLast_  =nbEvents_;
  if (nbEventsPerSec_.value_>0) {
    if (nbEventsPerSec_<nbEventsPerSecMin_) nbEventsPerSecMin_=nbEventsPerSec_;
    if (nbEventsPerSec_>nbEventsPerSecMax_) nbEventsPerSecMax_=nbEventsPerSec_;
  }
  nbEventsPerSecAvg_=nbEvents_/nbMeasurements_;

  // number of MB per second measurement
  nbMBPerSec_=9.53674e-07*resourceTable_->nbBytes();
  nbMBTot_.value_+=nbMBPerSec_;
  if (nbMBPerSec_.value_>0) {
    if (nbMBPerSec_<nbMBPerSecMin_) nbMBPerSecMin_=nbMBPerSec_;
    if (nbMBPerSec_>nbMBPerSecMax_) nbMBPerSecMax_=nbMBPerSec_;
  }
  nbMBPerSecAvg_=nbMBTot_/nbMeasurements_;

  gui_->unlockInfoSpaces();
  
  lock_.give();
}


//______________________________________________________________________________
void FUResourceBroker::initTimer()
{
  try {
    toolbox::task::getTimerFactory()->createTimer(sourceId_);
    toolbox::task::Timer *timer=toolbox::task::getTimerFactory()->getTimer(sourceId_);
    timer->stop();
  }
  catch (toolbox::task::exception::Exception& e) {
    LOG4CPLUS_WARN(log_,"FUResourceBroker::initTimer() failed.");
  }
}


//______________________________________________________________________________
void FUResourceBroker::startTimer()
{
  try {
    toolbox::task::Timer* timer=toolbox::task::getTimerFactory()->getTimer(sourceId_);
    toolbox::TimeInterval oneSec(1.);
    toolbox::TimeVal      startTime=toolbox::TimeVal::gettimeofday();
    timer->start();
    timer->scheduleAtFixedRate(startTime,this,oneSec,gui_->monInfoSpace(),sourceId_);
    
  }
  catch (toolbox::task::exception::Exception& e) {
    LOG4CPLUS_WARN(log_,"FUResourceBroker::startTimer() failed.");
  }
}


//______________________________________________________________________________
void FUResourceBroker::stopTimer()
{ 
  try {
    toolbox::task::Timer *timer=toolbox::task::getTimerFactory()->getTimer(sourceId_);
    timer->stop();
    toolbox::task::getTimerFactory()->removeTimer(sourceId_);
  }
  catch (toolbox::task::exception::Exception& e) {
    LOG4CPLUS_WARN(log_,"FUResourceBroker::stopTimer() failed.");
  }
}


//______________________________________________________________________________
void FUResourceBroker::actionPerformed(xdata::Event& e)
{
  if (0==resourceTable_) return;
  
  gui_->lockInfoSpaces();
  
  if (e.type()=="ItemRetrieveEvent") {
    
    string item=dynamic_cast<xdata::ItemRetrieveEvent&>(e).itemName();
    
    if (item=="nbShmClients")      nbShmClients_     =resourceTable_->nbShmClients();
    if (item=="nbAllocatedEvents") nbAllocatedEvents_=resourceTable_->nbAllocated();
    if (item=="nbPendingRequests") nbPendingRequests_=resourceTable_->nbPending();
    if (item=="nbReceivedEvents")  nbReceivedEvents_ =resourceTable_->nbCompleted();
    if (item=="nbProcessedEvents") nbProcessedEvents_=resourceTable_->nbProcessed();
    if (item=="nbDiscardedEvents") nbDiscardedEvents_=resourceTable_->nbDiscarded();
    if (item=="nbLostEvents")      nbLostEvents_     =resourceTable_->nbLost();
    if (item=="nbDataErrors")      nbDataErrors_     =resourceTable_->nbErrors();
    if (item=="nbCrcErrors")       nbCrcErrors_      =resourceTable_->nbCrcErrors();
    if (item=="nbAllocateSent")    nbAllocateSent_   =resourceTable_->nbAllocSent();
  }
  
  if (e.type()=="ItemChangedEvent") {
    
    string item=dynamic_cast<xdata::ItemChangedEvent&>(e).itemName();
    
    if (item=="doFedIdCheck") FUResource::doFedIdCheck(doFedIdCheck_);
    if (item=="doCrcCheck")   resourceTable_->setDoCrcCheck(doCrcCheck_);
    if (item=="runNumber") {
      resourceTable_->reset();
      gui_->resetCounters();
    }
  }
  
  gui_->unlockInfoSpaces();
}


//______________________________________________________________________________
void FUResourceBroker::connectToBUs()
{
  if (0!=bu_.size()) return;
  
  typedef set<xdaq::ApplicationDescriptor*> AppDescSet_t;
  typedef AppDescSet_t::iterator            AppDescIter_t;
    
  AppDescSet_t buAppDescs=
    getApplicationContext()->getDefaultZone()->
    getApplicationDescriptors(buClassName_.toString());
  
  UInt_t maxBuInstance(0);
  for (AppDescIter_t it=buAppDescs.begin();it!=buAppDescs.end();++it)
    if ((*it)->getInstance()>maxBuInstance) maxBuInstance=(*it)->getInstance();
  
  bu_.resize(maxBuInstance+1);
  bu_.assign(bu_.size(),0);
  
  bool buInstValid(false);
  
  for (UInt_t i=0;i<bu_.size();i++) {
    for (AppDescIter_t it=buAppDescs.begin();it!=buAppDescs.end();++it) {
      if (i==(*it)->getInstance()&&0==bu_[i]) {
	bu_[i]=new BUProxy(getApplicationDescriptor(),
			   *it, 
			   getApplicationContext(),
			   i2oPool_);
	if (i==buInstance_) buInstValid=true;
      }
    }
  }

  if (!buInstValid) LOG4CPLUS_ERROR(log_,"invalid buInstance! reset!!");
}


//______________________________________________________________________________
void FUResourceBroker::I2O_FU_TAKE_Callback(toolbox::mem::Reference* bufRef)
{
  // start the timer only upon receiving the first message
  if (nbTakeReceived_.value_==0) startTimer();
  
  nbTakeReceived_.value_++;

  bool eventComplete=resourceTable_->buildResource(bufRef);

  if (eventComplete&&doDropEvents_) {
      UInt_t evtNumber, buResourceId;
      FEDRawDataCollection *fedColl=
	resourceTable_->rqstEvent(evtNumber,buResourceId);
      resourceTable_->sendDiscard(buResourceId);
      if (!shmMode_) delete fedColl;
  }
  
}


//______________________________________________________________________________
void FUResourceBroker::webPageRequest(xgi::Input *in,xgi::Output *out)
  throw (xgi::exception::Exception)
{
  string name=in->getenv("PATH_INFO");
  if (name.empty()) name="defaultWebPage";
  static_cast<xgi::MethodSignature*>(gui_->getMethod(name))->invoke(in,out);
}


//______________________________________________________________________________
void FUResourceBroker::exportParameters()
{
  assert(0!=gui_);
  
  gui_->addMonitorParam("url",                &url_);
  gui_->addMonitorParam("class",              &class_);
  gui_->addMonitorParam("instance",           &instance_);
  gui_->addMonitorParam("runNumber",          &runNumber_);
  gui_->addMonitorParam("stateName",          &stateName_);
  gui_->addMonitorParam("nbShmClients",       &nbShmClients_);

  gui_->addMonitorParam("nbMBTot",            &nbMBTot_);
  gui_->addMonitorParam("nbMBPerSec",         &nbMBPerSec_);
  gui_->addMonitorParam("nbMBPerSecMin",      &nbMBPerSecMin_);
  gui_->addMonitorParam("nbMBPerSecMax",      &nbMBPerSecMax_);
  gui_->addMonitorParam("nbMBPerSecAvg",      &nbMBPerSecAvg_);

  gui_->addMonitorCounter("nbEvents",         &nbEvents_);
  gui_->addMonitorCounter("nbEventsPerSec",   &nbEventsPerSec_);
  gui_->addMonitorCounter("nbEventsPerSecMin",&nbEventsPerSecMin_);
  gui_->addMonitorCounter("nbEventsPerSecMax",&nbEventsPerSecMax_);
  gui_->addMonitorCounter("nbEventsPerSecAvg",&nbEventsPerSecAvg_);
  gui_->addMonitorCounter("nbAllocatedEvents",&nbAllocatedEvents_);
  gui_->addMonitorCounter("nbPendingRequests",&nbPendingRequests_);
  gui_->addMonitorCounter("nbReceivedEvents", &nbReceivedEvents_);
  gui_->addMonitorCounter("nbDiscardedEvents",&nbDiscardedEvents_);
  gui_->addMonitorCounter("nbProcessedEvents",&nbProcessedEvents_);
  gui_->addMonitorCounter("nbLostEvents",     &nbLostEvents_);
  gui_->addMonitorCounter("nbDataErrors",     &nbDataErrors_);
  gui_->addMonitorCounter("nbCrcErrors",      &nbCrcErrors_);

  gui_->addStandardParam("shmMode",           &shmMode_);
  gui_->addStandardParam("eventBufferSize",   &eventBufferSize_);
  //gui_->addStandardParam("doDumpLostEvents",   &doDumpLostEvents_);
  gui_->addStandardParam("doDropEvents",      &doDropEvents_);
  gui_->addStandardParam("doFedIdCheck",      &doFedIdCheck_);
  gui_->addStandardParam("doCrcCheck",        &doCrcCheck_);
  gui_->addStandardParam("buClassName",       &buClassName_);
  gui_->addStandardParam("buInstance",        &buInstance_);
  gui_->addStandardParam("queueSize",         &queueSize_);
  
  gui_->addDebugCounter("nbAllocateSent",     &nbAllocateSent_);
  gui_->addDebugCounter("nbTakeReceived",     &nbTakeReceived_);

  gui_->exportParameters();

  gui_->addItemRetrieveListener("nbShmClients",     this);
  gui_->addItemRetrieveListener("nbAllocatedEvents",this);
  gui_->addItemRetrieveListener("nbPendingRequests",this);
  gui_->addItemRetrieveListener("nbReceivedEvents", this);
  gui_->addItemRetrieveListener("nbProcessedEvents",this);
  gui_->addItemRetrieveListener("nbDiscardedEvents",this);
  gui_->addItemRetrieveListener("nbLostEvents",     this);
  gui_->addItemRetrieveListener("nbDataErrors",     this);
  gui_->addItemRetrieveListener("nbCrcErrors",      this);
  gui_->addItemRetrieveListener("nbAllocateSent",   this);
  
  gui_->addItemChangedListener("doFedIdCheck",      this);
  gui_->addItemChangedListener("doCrcCheck",        this);
  gui_->addItemChangedListener("runNumber",         this);
}


//______________________________________________________________________________
void FUResourceBroker::reset()
{
  gui_->resetCounters();
  
  nbMBTot_          =0.0;
  nbMBPerSec_       =0.0;
  nbMBPerSecMin_    =1e06;
  nbMBPerSecMax_    =0.0;
  nbMBPerSecAvg_    =0.0;
  nbEventsPerSecMin_=10000;
  
  nbMeasurements_   =0;
  nbEventsLast_     =0;
}


////////////////////////////////////////////////////////////////////////////////
// XDAQ instantiator implementation macro
////////////////////////////////////////////////////////////////////////////////

XDAQ_INSTANTIATOR_IMPL(FUResourceBroker)
