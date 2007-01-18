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

#include "xoap/include/xoap/SOAPEnvelope.h"
#include "xoap/include/xoap/SOAPBody.h"
#include "xoap/include/xoap/domutils.h"

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
  , lock_(BSem::FULL)
  , gui_(0)
  , log_(getApplicationLogger())
  , fsm_(0)
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
  , queueSize_(16)
  , nbAllocateSent_(0)
  , nbTakeReceived_(0)
  , nbMeasurements_(0)
  , nbEventsLast_(0)
{
  // set source id in evf::RunBase
  url_     =
    getApplicationDescriptor()->getContextDescriptor()->getURL()+"/"+
    getApplicationDescriptor()->getURN();
  class_   =getApplicationDescriptor()->getClassName();
  instance_=getApplicationDescriptor()->getInstance();
  
  sourceId_=class_.toString()+"_"+instance_.toString();
  RunBase::sourceId_=sourceId_;
  
  // initialize the finite state machine
  fsm_=new EPStateMachine(log_);
  fsm_->init<FUResourceBroker>(this);

  // i2o callback for FU_TAKE messages from builder unit
  i2o::bind(this,
	    &FUResourceBroker::I2O_FU_TAKE_Callback,
	    I2O_FU_TAKE,
	    XDAQ_ORGANIZATION_ID);
  
  // bind HyperDAQ web pages
  xgi::bind(this,&evf::FUResourceBroker::webPageRequest,"Default");
  gui_=new WebGUI(this,fsm_);
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
  toolbox::task::getTimerFactory()->createTimer(sourceId_);
  toolbox::task::Timer *timer=toolbox::task::getTimerFactory()->getTimer(sourceId_);
  timer->stop();
  cout<<"initTimer() finished."<<endl;
}


//______________________________________________________________________________
void FUResourceBroker::startTimer()
{
  toolbox::task::Timer *timer(0);
  try {
    timer=toolbox::task::getTimerFactory()->getTimer(sourceId_);
  }
  catch (toolbox::task::exception::Exception& e) {
    LOG4CPLUS_ERROR(log_,"getTimer() failed.");
  }

  if (0!=timer) {
    toolbox::TimeInterval oneSec(1.);
    toolbox::TimeVal      startTime=toolbox::TimeVal::gettimeofday();
    timer->start();
    timer->scheduleAtFixedRate(startTime,this,oneSec,gui_->monInfoSpace(),sourceId_);
  }
}


//______________________________________________________________________________
void FUResourceBroker::stopTimer()
{ 
  toolbox::task::Timer *timer=toolbox::task::getTimerFactory()->getTimer(sourceId_);
  if (0!=timer) timer->stop();
  toolbox::task::getTimerFactory()->removeTimer(sourceId_);
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
void FUResourceBroker::configureAction(toolbox::Event::Reference e) 
  throw (toolbox::fsm::exception::Exception)
{
  // establish connection to builder unit(s)
  connectToBUs();

  // initialize resource table
  if (0==resourceTable_) {
    resourceTable_=new FUResourceTable(queueSize_,eventBufferSize_,shmMode_,
				       bu_[buInstance_],log_);
    resourceTable_->resetCounters();
  }
  else if (resourceTable_->nbResources()!=queueSize_) {
    resourceTable_->initialize(queueSize_,eventBufferSize_);
  }
  else {
    resourceTable_->reset();
  }
  
  // reset counters and other variables to 'configured' state
  reset();
  
  // initialize timer
  initTimer();
  
  LOG4CPLUS_INFO(log_,"FUResourceBroker -> CONFIGURED <-");
}


//______________________________________________________________________________
void FUResourceBroker::enableAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  // request events from builder unit
  if (shmMode_) resourceTable_->startWorkLoop();
  resourceTable_->sendAllocate();
  
  LOG4CPLUS_INFO(log_,"FUResourceBroker -> ENABLED <-");
}


//______________________________________________________________________________
void FUResourceBroker::suspendAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  lock_.take();
  LOG4CPLUS_INFO(log_,"FUResourceBroker -> SUSPENDED <-");
}


//______________________________________________________________________________
void FUResourceBroker::resumeAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  lock_.give();
  LOG4CPLUS_INFO(log_,"FUResourceBroker -> RESUMED <-");
}


//______________________________________________________________________________
void FUResourceBroker::haltAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  stopTimer();
  
  UInt_t count = 0;
  while (count<10) {
    if (resourceTable_->nbShmClients()==0) {
      delete resourceTable_;
      resourceTable_=0;
      LOG4CPLUS_WARN(log_,++count<<". try to destroy resource table succeeded.");
      break;
    }
    else {
      ::sleep(1);
      count++;
      LOG4CPLUS_WARN(log_,count<<". try to destroy resource table failed.");
    }
  } 

  if (0!=resourceTable_)
    LOG4CPLUS_ERROR(log_,"Failed to destroy resource table.");
  
  LOG4CPLUS_INFO(log_,"FUResourceBroker -> HALTED <-");
}


//______________________________________________________________________________
void FUResourceBroker::nullAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  LOG4CPLUS_INFO(log_,"FUResourceBroker::nullAction() called.");
}


//______________________________________________________________________________
xoap::MessageReference FUResourceBroker::fireEvent(xoap::MessageReference msg)
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
      return fsm_->processFSMCommand(commandName);
    }
  }
  XCEPT_RAISE(xoap::exception::Exception,"Command not found");
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
  
  if (bu_.size()!=buAppDescs.size()) LOG4CPLUS_ERROR(log_,"maxBuInstance > #BUs!");
  
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
  //if (fsm_->getCurrentState()!='E') {
  //bufRef->release();
  //return;
  //}

  // start the timer only upon receiving the first message
  if (nbTakeReceived_.value_==0) startTimer();
  
  nbTakeReceived_.value_++;

  bool eventComplete=resourceTable_->buildResource(bufRef);

  if (eventComplete&&doDropEvents_) {
      UInt_t evtNumber;
      UInt_t buResourceId;
      FEDRawDataCollection *fedColl=
	resourceTable_->rqstEvent(evtNumber,buResourceId);
      resourceTable_->sendDiscard(buResourceId);
      if (!shmMode_) {
	cout<<"delete fedColl."<<endl;
	delete fedColl;
      }
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
  gui_->addMonitorParam("stateName",          &fsm_->stateName_);
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
