///////////////////////////////////////////////////////////////////////////////
//
// FUResourceBroker
// ----------------
//
//            10/20/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ResourceBroker/interface/FUResourceBroker.h"
#include "EventFilter/ResourceBroker/interface/BUProxy.h"
#include "EventFilter/ResourceBroker/interface/SMProxy.h"

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
  , fsm_(this)
  , lock_(BSem::FULL)
  , gui_(0)
  , log_(getApplicationLogger())
  , bu_(0)
  , sm_(0)
  , i2oPool_(0)
  , resourceTable_(0)
  , instance_(0)
  , runNumber_(0)
  , nbShmClients_(0)
  , acceptRate_(0.0)
  , nbMBInput_(0.0)
  , nbMBInputPerSec_(0.0)
  , nbMBInputPerSecMin_(0.0)
  , nbMBInputPerSecMax_(0.0)
  , nbMBInputPerSecAvg_(0.0)
  , nbMBOutput_(0.0)
  , nbMBOutputPerSec_(0.0)
  , nbMBOutputPerSecMin_(0.0)
  , nbMBOutputPerSecMax_(0.0)
  , nbMBOutputPerSecAvg_(0.0)
  , nbInputEvents_(0)
  , nbInputEventsPerSec_(0)
  , nbInputEventsPerSecMin_(0)
  , nbInputEventsPerSecMax_(0)
  , nbInputEventsPerSecAvg_(0)
  , nbOutputEvents_(0)
  , nbOutputEventsPerSec_(0)
  , nbOutputEventsPerSecMin_(0)
  , nbOutputEventsPerSecMax_(0)
  , nbOutputEventsPerSecAvg_(0)
  , nbAllocatedEvents_(0)
  , nbPendingRequests_(0)
  , nbProcessedEvents_(0)
  , nbAcceptedEvents_(0)
  , nbDiscardedEvents_(0)
  , nbLostEvents_(0)
  , nbDataErrors_(0)
  , nbCrcErrors_(0)
  , segmentationMode_(false)
  , nbRawCells_(32)
  , nbRecoCells_(0)
  , nbDqmCells_(0)
  , rawCellSize_(4194304)  // 4MB
  , recoCellSize_(4194304) // 4MB
  , dqmCellSize_(4194304)  // 4MB
  , doDropEvents_(false)
  , doFedIdCheck_(true)
  , doCrcCheck_(1)
  , buClassName_("BU")
  , buInstance_(0)
  , smClassName_("StorageManager")
  , smInstance_(0)
  , nbAllocateSent_(0)
  , nbTakeReceived_(0)
  , nbDataDiscardReceived_(0)
  , nbDqmDiscardReceived_(0)
  , nbMeasurements_(0)
  , nbInputEventsLast_(0)
  , nbOutputEventsLast_(0)
{
  // setup finite state machine (binding relevant callbacks)
  fsm_.initialize<evf::FUResourceBroker>(this);
  
  // set url, class, instance, and sourceId (=class_instance)
  url_     =
    getApplicationDescriptor()->getContextDescriptor()->getURL()+"/"+
    getApplicationDescriptor()->getURN();
  class_   =getApplicationDescriptor()->getClassName();
  instance_=getApplicationDescriptor()->getInstance();
  sourceId_=class_.toString()+"_"+instance_.toString();
  
  // bind i2o callbacks
  i2o::bind(this,&FUResourceBroker::I2O_FU_TAKE_Callback,
	    I2O_FU_TAKE,XDAQ_ORGANIZATION_ID);
  i2o::bind(this,&FUResourceBroker::I2O_FU_DATA_DISCARD_Callback,
	    I2O_FU_DATA_DISCARD,XDAQ_ORGANIZATION_ID);
  i2o::bind(this,&FUResourceBroker::I2O_FU_DQM_DISCARD_Callback,
	    I2O_FU_DQM_DISCARD,XDAQ_ORGANIZATION_ID);
  
  
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
bool FUResourceBroker::configuring(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(log_, "Start configuring ...");
    connectToBUandSM();
    resourceTable_=new FUResourceTable(segmentationMode_.value_,
				       nbRawCells_.value_,
				       nbRecoCells_.value_,
				       nbDqmCells_.value_,
				       rawCellSize_.value_,
				       recoCellSize_.value_,
				       dqmCellSize_.value_,
				       bu_,sm_,
				       log_);
    resourceTable_->resetCounters();
    reset();
    LOG4CPLUS_INFO(log_, "Finished configuring!");
    
    fsm_.fireEvent("ConfigureDone",this);
  }
  catch (xcept::Exception &e) {
    std::string msg = "configuring FAILED: " + (string)e.what();
    fsm_.fireFailed(msg,this);
  }
  
  return false;
}


//______________________________________________________________________________
bool FUResourceBroker::enabling(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(log_, "Start enabling ...");
    initTimer();
    resourceTable_->startDiscardWorkLoop();
    resourceTable_->sendAllocate();
    LOG4CPLUS_INFO(log_, "Finished enabling!");
    fsm_.fireEvent("EnableDone",this);
  }
  catch (xcept::Exception &e) {
    std::string msg = "enabling FAILED: " + (string)e.what();
    fsm_.fireFailed(msg,this);
  }
  
  return false;
}


//______________________________________________________________________________
bool FUResourceBroker::stopping(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(log_, "Start stopping :) ...");
    resourceTable_->shutDownClients();
    stopTimer();
    LOG4CPLUS_INFO(log_, "Finished stopping!");
    fsm_.fireEvent("StopDone",this);
  }
  catch (xcept::Exception &e) {
    std::string msg = "stopping FAILED: " + (string)e.what();
    fsm_.fireFailed(msg,this);
  }
  
  return false;
}


//______________________________________________________________________________
bool FUResourceBroker::halting(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(log_, "Start halting ...");
    resourceTable_->shutDownClients();
    stopTimer();
    UInt_t count = 0;
    while (count<10) {
      if (resourceTable_->isReadyToShutDown()) {
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
    
    fsm_.fireEvent("HaltDone",this);
  }
  catch (xcept::Exception &e) {
    std::string msg = "halting FAILED: " + (string)e.what();
    fsm_.fireFailed(msg,this);
  }
  
  return false;
}


//______________________________________________________________________________
xoap::MessageReference FUResourceBroker::fsmCallback(xoap::MessageReference msg)
  throw (xoap::exception::Exception)
{
  return fsm_.commandCallback(msg);
}


//______________________________________________________________________________
void FUResourceBroker::I2O_FU_TAKE_Callback(toolbox::mem::Reference* bufRef)
{
  if (nbTakeReceived_.value_==0) startTimer();
  nbTakeReceived_.value_++;
  bool eventComplete=resourceTable_->buildResource(bufRef);
  if (eventComplete&&doDropEvents_) resourceTable_->dropEvent();
}


//______________________________________________________________________________
void FUResourceBroker::I2O_FU_DATA_DISCARD_Callback(toolbox::mem::Reference* bufRef)
{
  nbDataDiscardReceived_.value_++;
  resourceTable_->discardDataEvent(bufRef);
}


//______________________________________________________________________________
void FUResourceBroker::I2O_FU_DQM_DISCARD_Callback(toolbox::mem::Reference* bufRef)
{
  nbDqmDiscardReceived_.value_++;
  resourceTable_->discardDqmEvent(bufRef);
}


//______________________________________________________________________________
void FUResourceBroker::connectToBUandSM() throw (evf::Exception)
{
  typedef set<xdaq::ApplicationDescriptor*> AppDescSet_t;
  typedef AppDescSet_t::iterator            AppDescIter_t;
  
  // locate input BU
  AppDescSet_t setOfBUs=
    getApplicationContext()->getDefaultZone()->
    getApplicationDescriptors(buClassName_.toString());
  
  if (0!=bu_) { delete bu_; bu_=0; }
  
  for (AppDescIter_t it=setOfBUs.begin();it!=setOfBUs.end();++it)
    if ((*it)->getInstance()==buInstance_)
      bu_=new BUProxy(getApplicationDescriptor(),*it,
		      getApplicationContext(),i2oPool_);
  
  if (0==bu_) {
    string msg=sourceId_+" failed to locate input BU!";
    XCEPT_RAISE(evf::Exception,msg);
  }
  
  // locate output SM
  AppDescSet_t setOfSMs=
    getApplicationContext()->getDefaultZone()->
    getApplicationDescriptors(smClassName_.toString());
  
  if (0!=sm_) { delete sm_; sm_=0; }
  
  for (AppDescIter_t it=setOfSMs.begin();it!=setOfSMs.end();++it)
    if ((*it)->getInstance()==smInstance_)
      sm_=new SMProxy(getApplicationDescriptor(),*it,
		      getApplicationContext(),i2oPool_);
  
  if (0==sm_) LOG4CPLUS_WARN(log_,sourceId_<<" failed to locate output SM!");
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
void FUResourceBroker::timeExpired(toolbox::task::TimerEvent& e)
{
  lock_.take();

  gui_->lockInfoSpaces();

  nbMeasurements_++;
 
  // number of input events per second measurement
  nbInputEvents_      =resourceTable_->nbCompleted();
  nbInputEventsPerSec_=nbInputEvents_-nbInputEventsLast_;
  nbInputEventsLast_  =nbInputEvents_;
  if (nbInputEventsPerSec_.value_>0) {
    if (nbInputEventsPerSec_<nbInputEventsPerSecMin_)
      nbInputEventsPerSecMin_=nbInputEventsPerSec_;
    if (nbInputEventsPerSec_>nbInputEventsPerSecMax_)
      nbInputEventsPerSecMax_=nbInputEventsPerSec_;
  }
  nbInputEventsPerSecAvg_=nbInputEvents_/nbMeasurements_;

  // number of MB per second measurement
  nbMBInputPerSec_=9.53674e-07*resourceTable_->nbBytesReceived();
  nbMBInput_.value_+=nbMBInputPerSec_;
  if (nbMBInputPerSec_.value_>0) {
    if (nbMBInputPerSec_<nbMBInputPerSecMin_)
      nbMBInputPerSecMin_=nbMBInputPerSec_;
    if (nbMBInputPerSec_>nbMBInputPerSecMax_)
      nbMBInputPerSecMax_=nbMBInputPerSec_;
  }
  nbMBInputPerSecAvg_=nbMBInput_/nbMeasurements_;

  // number of output events per second measurement
  nbOutputEvents_      =resourceTable_->nbSent();
  nbOutputEventsPerSec_=nbOutputEvents_-nbOutputEventsLast_;
  nbOutputEventsLast_  =nbOutputEvents_;
  if (nbOutputEventsPerSec_.value_>0) {
    if (nbOutputEventsPerSec_<nbOutputEventsPerSecMin_)
      nbOutputEventsPerSecMin_=nbOutputEventsPerSec_;
    if (nbOutputEventsPerSec_>nbOutputEventsPerSecMax_)
      nbOutputEventsPerSecMax_=nbOutputEventsPerSec_;
  }
  nbOutputEventsPerSecAvg_=nbOutputEvents_/nbMeasurements_;

  // number of MB per second measurement
  nbMBOutputPerSec_=9.53674e-07*resourceTable_->nbBytesSent();
  nbMBOutput_.value_+=nbMBOutputPerSec_;
  if (nbMBOutputPerSec_.value_>0) {
    if (nbMBOutputPerSec_<nbMBOutputPerSecMin_)
      nbMBOutputPerSecMin_=nbMBOutputPerSec_;
    if (nbMBOutputPerSec_>nbMBOutputPerSecMax_)
      nbMBOutputPerSecMax_=nbMBOutputPerSec_;
  }
  nbMBOutputPerSecAvg_=nbMBOutput_/nbMeasurements_;
  
  // accept rate
  acceptRate_=nbOutputEvents_/nbInputEvents_;
  
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
  catch (xcept::Exception& e) {
    LOG4CPLUS_WARN(log_,"FUResourceBroker::initTimer() failed: "<<e.what());
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
  catch (xcept::Exception& e) {
    LOG4CPLUS_WARN(log_,"FUResourceBroker::startTimer() failed: "<<e.what());
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
  catch (xcept::Exception& e) {
    LOG4CPLUS_WARN(log_,"FUResourceBroker::stopTimer() failed: "<<e.what());
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
    if (item=="nbProcessedEvents") nbProcessedEvents_=resourceTable_->nbProcessed();
    if (item=="nbAcceptedEvents")  nbAcceptedEvents_ =resourceTable_->nbAccepted();
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
void FUResourceBroker::exportParameters()
{
  assert(0!=gui_);
  
  gui_->addMonitorParam("url",                      &url_);
  gui_->addMonitorParam("class",                    &class_);
  gui_->addMonitorParam("instance",                 &instance_);
  gui_->addMonitorParam("runNumber",                &runNumber_);
  gui_->addMonitorParam("stateName",                 fsm_.stateName());
  gui_->addMonitorParam("nbShmClients",             &nbShmClients_);

  gui_->addMonitorParam("acceptRate",               &acceptRate_);
  
  gui_->addMonitorParam("nbMBInput",                &nbMBInput_);
  gui_->addMonitorParam("nbMBInputPerSec",          &nbMBInputPerSec_);
  gui_->addDebugParam("nbMBInputPerSecMin",         &nbMBInputPerSecMin_);
  gui_->addDebugParam("nbMBInputPerSecMax",         &nbMBInputPerSecMax_);
  gui_->addMonitorParam("nbMBInputPerSecAvg",       &nbMBInputPerSecAvg_);

  gui_->addMonitorParam("nbMBOutput",               &nbMBOutput_);
  gui_->addMonitorParam("nbMBOutputPerSec",         &nbMBOutputPerSec_);
  gui_->addDebugParam("nbMBOutputPerSecMin",        &nbMBOutputPerSecMin_);
  gui_->addDebugParam("nbMBOutputPerSecMax",        &nbMBOutputPerSecMax_);
  gui_->addMonitorParam("nbMBOutputPerSecAvg",      &nbMBOutputPerSecAvg_);

  gui_->addMonitorCounter("nbInputEvents",          &nbInputEvents_);
  gui_->addMonitorCounter("nbInputEventsPerSec",    &nbInputEventsPerSec_);
  gui_->addDebugCounter("nbInputEventsPerSecMin",   &nbInputEventsPerSecMin_);
  gui_->addDebugCounter("nbInputEventsPerSecMax",   &nbInputEventsPerSecMax_);
  gui_->addMonitorCounter("nbInputEventsPerSecAvg", &nbInputEventsPerSecAvg_);

  gui_->addMonitorCounter("nbOutputEvents",         &nbOutputEvents_);
  gui_->addMonitorCounter("nbOutputEventsPerSec",   &nbOutputEventsPerSec_);
  gui_->addDebugCounter("nbOutputEventsPerSecMin",  &nbOutputEventsPerSecMin_);
  gui_->addDebugCounter("nbOutputEventsPerSecMax",  &nbOutputEventsPerSecMax_);
  gui_->addMonitorCounter("nbOutputEventsPerSecAvg",&nbOutputEventsPerSecAvg_);

  gui_->addMonitorCounter("nbAllocatedEvents",      &nbAllocatedEvents_);
  gui_->addMonitorCounter("nbPendingRequests",      &nbPendingRequests_);
  gui_->addMonitorCounter("nbProcessedEvents",      &nbProcessedEvents_);
  gui_->addMonitorCounter("nbAcceptedEvents",       &nbAcceptedEvents_);
  gui_->addMonitorCounter("nbDiscardedEvents",      &nbDiscardedEvents_);
  gui_->addMonitorCounter("nbLostEvents",           &nbLostEvents_);
  gui_->addMonitorCounter("nbDataErrors",           &nbDataErrors_);
  gui_->addMonitorCounter("nbCrcErrors",            &nbCrcErrors_);

  gui_->addStandardParam("segmentationMode",        &segmentationMode_);
  gui_->addStandardParam("nbRawCells",              &nbRawCells_);
  gui_->addStandardParam("nbRecoCells",             &nbRecoCells_);
  gui_->addStandardParam("nbDqmCells",              &nbDqmCells_);
  gui_->addStandardParam("rawCellSize",             &rawCellSize_);
  gui_->addStandardParam("recoCellSize",            &recoCellSize_);
  gui_->addStandardParam("dqmCellSize",             &dqmCellSize_);

  gui_->addStandardParam("doDropEvents",            &doDropEvents_);
  gui_->addStandardParam("doFedIdCheck",            &doFedIdCheck_);
  gui_->addStandardParam("doCrcCheck",              &doCrcCheck_);
  gui_->addStandardParam("buClassName",             &buClassName_);
  gui_->addStandardParam("buInstance",              &buInstance_);
  gui_->addStandardParam("smClassName",             &smClassName_);
  gui_->addStandardParam("smInstance",              &smInstance_);
  gui_->addStandardParam("foundRcmsStateListener",   fsm_.foundRcmsStateListener());

  gui_->addDebugCounter("nbAllocateSent",           &nbAllocateSent_);
  gui_->addDebugCounter("nbTakeReceived",           &nbTakeReceived_);
  gui_->addDebugCounter("nbDataDiscardReceived",    &nbDataDiscardReceived_);
  gui_->addDebugCounter("nbDqmDiscardReceived",     &nbDqmDiscardReceived_);

  gui_->exportParameters();

  gui_->addItemRetrieveListener("nbShmClients",     this);
  gui_->addItemRetrieveListener("nbAllocatedEvents",this);
  gui_->addItemRetrieveListener("nbPendingRequests",this);
  gui_->addItemRetrieveListener("nbProcessedEvents",this);
  gui_->addItemRetrieveListener("nbAcceptedEvents", this);
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
  
  nbMBInput_              =  0.0;
  nbMBInputPerSec_        =  0.0;
  nbMBInputPerSecMin_     = 1e06;
  nbMBInputPerSecMax_     =  0.0;
  nbMBInputPerSecAvg_     =  0.0;
  nbInputEventsPerSecMin_ =10000;

  nbMBOutput_             =  0.0;
  nbMBOutputPerSec_       =  0.0;
  nbMBOutputPerSecMin_    = 1e06;
  nbMBOutputPerSecMax_    =  0.0;
  nbMBOutputPerSecAvg_    =  0.0;
  nbOutputEventsPerSecMin_=10000;
  
  nbMeasurements_         =    0;
  nbInputEventsLast_      =    0;
  nbOutputEventsLast_     =    0;
}


////////////////////////////////////////////////////////////////////////////////
// XDAQ instantiator implementation macro
////////////////////////////////////////////////////////////////////////////////

XDAQ_INSTANTIATOR_IMPL(FUResourceBroker)
