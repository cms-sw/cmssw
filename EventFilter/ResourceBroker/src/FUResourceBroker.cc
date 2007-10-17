////////////////////////////////////////////////////////////////////////////////
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

#include "i2o/Method.h"
#include "interface/shared/i2oXFunctionCodes.h"
#include "xcept/tools.h"

#include "toolbox/mem/HeapAllocator.h"
#include "toolbox/mem/Reference.h"
#include "toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/mem/exception/Exception.h"

#include "xoap/MessageReference.h"
#include "xoap/MessageFactory.h"
#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPBody.h"
#include "xoap/domutils.h"
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
  , gui_(0)
  , log_(getApplicationLogger())
  , bu_(0)
  , sm_(0)
  , i2oPool_(0)
  , resourceTable_(0)
  , wlMonitoring_(0)
  , asMonitoring_(0)
  , instance_(0)
  , runNumber_(0)
  , nbShmClients_(0)
  , deltaT_(0.0)
  , deltaNbInput_(0)
  , deltaNbOutput_(0)
  , deltaInputSumOfSquares_(0)
  , deltaOutputSumOfSquares_(0)
  , deltaInputSumOfSizes_(0)
  , deltaOutputSumOfSizes_(0)
  , ratio_(0.0)
  , inputThroughput_(0.0)
  , inputRate_(0.0)
  , inputAverage_(0.0)
  , inputRms_(0.0)
  , outputThroughput_(0.0)
  , outputRate_(0.0)
  , outputAverage_(0.0)
  , outputRms_(0.0)
  , nbAllocatedEvents_(0)
  , nbPendingRequests_(0)
  , nbReceivedEvents_(0)
  , nbAcceptedEvents_(0)
  , nbSentEvents_(0)
  , nbDiscardedEvents_(0)
  , nbLostEvents_(0)
  , nbDataErrors_(0)
  , nbCrcErrors_(0)
  , segmentationMode_(false)
  , nbRawCells_(32)
  , nbRecoCells_(8)
  , nbDqmCells_(8)
  , rawCellSize_(0x400000)  // 4MB
  , recoCellSize_(0x800000) // 8MB
  , dqmCellSize_(0x800000)  // 8MB
  , doDropEvents_(false)
  , doFedIdCheck_(true)
  , doCrcCheck_(1)
  , doDumpEvents_(0)
  , buClassName_("BU")
  , buInstance_(0)
  , smClassName_("StorageManager")
  , smInstance_(0)
  , monSleepSec_(1)
  , nbAllocateSent_(0)
  , nbTakeReceived_(0)
  , nbDataDiscardReceived_(0)
  , nbDqmDiscardReceived_(0)
  , nbInputLast_(0)
  , nbInputLastSumOfSquares_(0)
  , nbInputLastSumOfSizes_(0)
  , nbOutputLast_(0)
  , nbOutputLastSumOfSquares_(0)
  , nbOutputLastSumOfSizes_(0)
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
    resourceTable_->setDoCrcCheck(doCrcCheck_);
    resourceTable_->setDoDumpEvents(doDumpEvents_);
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
    startMonitoringWorkLoop();
    resourceTable_->resetCounters();
    resourceTable_->startDiscardWorkLoop();
    resourceTable_->startSendDataWorkLoop();
    resourceTable_->startSendDqmWorkLoop();
    resourceTable_->sendAllocate();
    LOG4CPLUS_INFO(log_, "Finished enabling!");
    fsm_.fireEvent("EnableDone",this);
  }
  catch (xcept::Exception &e) {
    std::string msg="enabling FAILED: "+xcept::stdformat_exception_history(e);
    fsm_.fireFailed(msg,this);
  }
  
  return false;
}


//______________________________________________________________________________
bool FUResourceBroker::stopping(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(log_, "Start stopping :) ...");
    resourceTable_->stop();
    UInt_t count = 0;
    while (count<10) {
      if (resourceTable_->isReadyToShutDown()) {
	LOG4CPLUS_INFO(log_,"ResourceTable successfully shutdown ("<<count+1<<").");
	break;
      }
      else {
	LOG4CPLUS_DEBUG(log_,"Waiting for ResourceTable to shutdown ("<<++count<<")");
	::sleep(1);
      }
    }
    
    if (count<10) {
      LOG4CPLUS_INFO(log_, "Finished stopping!");
      fsm_.fireEvent("StopDone",this);
    }
    else {
      std::string msg = "stopping FAILED: ResourceTable shutdown timed out.";
      fsm_.fireFailed(msg,this);
    }
  }
  catch (xcept::Exception &e) {
    std::string msg = "stopping FAILED: "+xcept::stdformat_exception_history(e);
    fsm_.fireFailed(msg,this);
  }
  
  return false;
}


//______________________________________________________________________________
bool FUResourceBroker::halting(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(log_, "Start halting ...");
    if (resourceTable_->isActive()) {
      resourceTable_->halt();
      UInt_t count = 0;
      while (count<10) {
	if (resourceTable_->isReadyToShutDown()) {
	  delete resourceTable_;
	  resourceTable_=0;
	  LOG4CPLUS_INFO(log_,count+1<<". try to destroy resource table succeeded!");
	  break;
	}
	else {
	  LOG4CPLUS_DEBUG(log_,++count<<". try to destroy resource table failed ...");
	  ::sleep(1);
	}
      }
    }
    else {
      delete resourceTable_;
      resourceTable_=0;
    }
    
    if (0==resourceTable_) {
      LOG4CPLUS_INFO(log_,"Finished halting!");
      fsm_.fireEvent("HaltDone",this);
    }
    else {
      std::string msg = "halting FAILED: ResourceTable shutdown timed out.";
      fsm_.fireFailed(msg,this);
    }
  }
  catch (xcept::Exception &e) {
    std::string msg = "halting FAILED: "+xcept::stdformat_exception_history(e);
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
void FUResourceBroker::actionPerformed(xdata::Event& e)
{
  if (0==resourceTable_) return;
  
  gui_->monInfoSpace()->lock();

  if (e.type()=="urn:xdata-event:ItemGroupRetrieveEvent") {
    nbShmClients_     =resourceTable_->nbShmClients();
    nbAllocatedEvents_=resourceTable_->nbAllocated();
    nbPendingRequests_=resourceTable_->nbPending();
    nbReceivedEvents_ =resourceTable_->nbCompleted();
    nbAcceptedEvents_ =resourceTable_->nbAccepted();
    nbSentEvents_     =resourceTable_->nbSent();
    nbDiscardedEvents_=resourceTable_->nbDiscarded();
    nbLostEvents_     =resourceTable_->nbLost();
    nbDataErrors_     =resourceTable_->nbErrors();
    nbCrcErrors_      =resourceTable_->nbCrcErrors();
    nbAllocateSent_   =resourceTable_->nbAllocSent();
  }
  else if (e.type()=="ItemChangedEvent") {
    
    string item=dynamic_cast<xdata::ItemChangedEvent&>(e).itemName();
    
    if (item=="doFedIdCheck") FUResource::doFedIdCheck(doFedIdCheck_);
    if (item=="doCrcCheck")   resourceTable_->setDoCrcCheck(doCrcCheck_);
    if (item=="doDumpEvents") resourceTable_->setDoDumpEvents(doDumpEvents_);
    if (item=="runNumber")    resourceTable_->resetCounters();
  }
  
  gui_->monInfoSpace()->unlock();
}


//______________________________________________________________________________
void FUResourceBroker::startMonitoringWorkLoop() throw (evf::Exception)
{
  struct timezone timezone;
  gettimeofday(&monStartTime_,&timezone);
  
  try {
    wlMonitoring_=
      toolbox::task::getWorkLoopFactory()->getWorkLoop(sourceId_+"Monitoring",
						       "waiting");
    if (!wlMonitoring_->isActive()) wlMonitoring_->activate();
    asMonitoring_=toolbox::task::bind(this,&FUResourceBroker::monitoring,
				      sourceId_+"Monitoring");
    wlMonitoring_->submit(asMonitoring_);
  }
  catch (xcept::Exception& e) {
    string msg = "Failed to start workloop 'Monitoring'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
}


//______________________________________________________________________________
bool FUResourceBroker::monitoring(toolbox::task::WorkLoop* wl)
{
  if (0==resourceTable_) return false;
  
  struct timeval  monEndTime;
  struct timezone timezone;
  
  gettimeofday(&monEndTime,&timezone);
  
  unsigned int nbInput             =resourceTable_->nbCompleted();
  unsigned int nbProcessed         =resourceTable_->nbDiscarded();
  uint64_t     nbInputSumOfSquares =resourceTable_->inputSumOfSquares();
  unsigned int nbInputSumOfSizes   =resourceTable_->inputSumOfSizes();
  unsigned int nbOutput            =resourceTable_->nbSent();
  uint64_t     nbOutputSumOfSquares=resourceTable_->outputSumOfSquares();
  unsigned int nbOutputSumOfSizes  =resourceTable_->outputSumOfSizes();
  uint64_t     deltaInputSumOfSquares;
  uint64_t     deltaOutputSumOfSquares;
  
  gui_->monInfoSpace()->lock();
  
  deltaT_.value_=deltaT(&monStartTime_,&monEndTime);
  monStartTime_=monEndTime;
  
  deltaNbInput_.value_=nbInput-nbInputLast_;
  nbInputLast_=nbInput;

  deltaNbOutput_.value_=nbOutput-nbOutputLast_;
  nbOutputLast_=nbOutput;
  
  deltaInputSumOfSquares=nbInputSumOfSquares-nbInputLastSumOfSquares_;
  deltaInputSumOfSquares_.value_=(double)deltaInputSumOfSquares;
  nbInputLastSumOfSquares_=nbInputSumOfSquares;

  deltaOutputSumOfSquares=nbOutputSumOfSquares-nbOutputLastSumOfSquares_;
  deltaOutputSumOfSquares_.value_=(double)deltaOutputSumOfSquares;
  nbOutputLastSumOfSquares_=nbOutputSumOfSquares;
  
  deltaInputSumOfSizes_.value_=nbInputSumOfSizes-nbInputLastSumOfSizes_;
  nbInputLastSumOfSizes_=nbInputSumOfSizes;

  deltaOutputSumOfSizes_.value_=nbOutputSumOfSizes-nbOutputLastSumOfSizes_;
  nbOutputLastSumOfSizes_=nbOutputSumOfSizes;
  
  //gui_->monInfoSpace()->unlock();
  
  if (nbProcessed!=0)
    ratio_=(double)nbOutput/(double)nbProcessed;
  else
    ratio_=0.0;
  
  if (deltaT_.value_!=0) {
    inputThroughput_ =deltaInputSumOfSizes_.value_/deltaT_.value_;
    outputThroughput_=deltaOutputSumOfSizes_.value_/deltaT_.value_;
    inputRate_       =deltaNbInput_.value_/deltaT_.value_;
    outputRate_      =deltaNbOutput_.value_/deltaT_.value_;
  }
  else {
    inputThroughput_ =0.0;
    outputThroughput_=0.0;
    inputRate_       =0.0;
    outputRate_      =0.0;
  }
  
  double meanOfSquares,mean,squareOfMean,variance;
  
  if(deltaNbInput_.value_!=0) {
    meanOfSquares=deltaInputSumOfSquares_.value_/((double)(deltaNbInput_.value_));
    mean=((double)(deltaInputSumOfSizes_.value_))/((double)(deltaNbInput_.value_));
    squareOfMean=mean*mean;
    variance=meanOfSquares-squareOfMean; if(variance<0.0) variance=0.0;
    
    inputAverage_=deltaInputSumOfSizes_.value_/deltaNbInput_.value_;
    inputRms_    =std::sqrt(variance);
  }
  else {
    inputAverage_=0.0;
    inputRms_    =0.0;
  }
  
  if(deltaNbOutput_.value_!=0) {
    meanOfSquares=deltaOutputSumOfSquares_.value_/((double)(deltaNbOutput_.value_));
    mean=((double)(deltaOutputSumOfSizes_.value_))/((double)(deltaNbOutput_.value_));
    squareOfMean=mean*mean;
    variance=meanOfSquares-squareOfMean; if(variance<0.0) variance=0.0;

    outputAverage_=deltaOutputSumOfSizes_.value_/deltaNbOutput_.value_;
    outputRms_    =std::sqrt(variance);
  }
  else {
    outputAverage_=0.0;
    outputRms_    =0.0;
  }

  gui_->monInfoSpace()->unlock();  

  ::sleep(monSleepSec_.value_);
  
  return true;
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

  gui_->addMonitorParam("deltaT",                   &deltaT_);
  gui_->addMonitorParam("deltaNbInput",             &deltaNbInput_);
  gui_->addMonitorParam("deltaNbOutput",            &deltaNbOutput_);
  gui_->addMonitorParam("deltaInputSumOfSquares",   &deltaInputSumOfSquares_);
  gui_->addMonitorParam("deltaOutputSumOfSquares",  &deltaOutputSumOfSquares_);
  gui_->addMonitorParam("deltaInputSumOfSizes",     &deltaInputSumOfSizes_);
  gui_->addMonitorParam("deltaOutputSumOfSizes",    &deltaOutputSumOfSizes_);
    
  gui_->addMonitorParam("ratio",                    &ratio_);
  gui_->addMonitorParam("inputThroughput",          &inputThroughput_);
  gui_->addMonitorParam("inputRate",                &inputRate_);
  gui_->addMonitorParam("inputAverage",             &inputAverage_);
  gui_->addMonitorParam("inputRms",                 &inputRms_);
  gui_->addMonitorParam("outputThroughput",         &outputThroughput_);
  gui_->addMonitorParam("outputRate",               &outputRate_);
  gui_->addMonitorParam("outputAverage",            &outputAverage_);
  gui_->addMonitorParam("outputRms",                &outputRms_);
  
  gui_->addMonitorCounter("nbAllocatedEvents",      &nbAllocatedEvents_);
  gui_->addMonitorCounter("nbPendingRequests",      &nbPendingRequests_);
  gui_->addMonitorCounter("nbReceivedEvents",       &nbReceivedEvents_);
  gui_->addMonitorCounter("nbAcceptedEvents",       &nbAcceptedEvents_);
  gui_->addMonitorCounter("nbSentEvents",           &nbSentEvents_);
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
  gui_->addStandardParam("doDumpEvents",            &doDumpEvents_);
  gui_->addStandardParam("buClassName",             &buClassName_);
  gui_->addStandardParam("buInstance",              &buInstance_);
  gui_->addStandardParam("smClassName",             &smClassName_);
  gui_->addStandardParam("smInstance",              &smInstance_);
  gui_->addStandardParam("monSleepSec",             &monSleepSec_);
  gui_->addStandardParam("foundRcmsStateListener",   fsm_.foundRcmsStateListener());

  gui_->addDebugCounter("nbAllocateSent",           &nbAllocateSent_);
  gui_->addDebugCounter("nbTakeReceived",           &nbTakeReceived_);
  gui_->addDebugCounter("nbDataDiscardReceived",    &nbDataDiscardReceived_);
  gui_->addDebugCounter("nbDqmDiscardReceived",     &nbDqmDiscardReceived_);

  gui_->exportParameters();

  gui_->addItemChangedListener("doFedIdCheck",      this);
  gui_->addItemChangedListener("doCrcCheck",        this);
  gui_->addItemChangedListener("doDumpEvents",      this);
  gui_->addItemChangedListener("runNumber",         this);
}


//______________________________________________________________________________
void FUResourceBroker::reset()
{
  gui_->resetCounters();
  
  deltaT_                  =0.0;
  deltaNbInput_            =  0;
  deltaNbOutput_           =  0;
  deltaInputSumOfSquares_  =0.0;
  deltaOutputSumOfSquares_ =0.0;
  deltaInputSumOfSizes_    =  0;
  deltaOutputSumOfSizes_   =  0;
  
  ratio_                   =0.0;
  inputThroughput_         =0.0;
  inputRate_               =0.0;
  inputAverage_            =0.0;
  inputRms_                =0.0;
  outputThroughput_        =0.0;
  outputRate_              =0.0;
  outputAverage_           =0.0;
  outputRms_               =0.0;
  
  nbInputLast_             =  0;
  nbInputLastSumOfSquares_ =  0;
  nbInputLastSumOfSizes_   =  0;
  nbOutputLast_            =  0;
  nbOutputLastSumOfSquares_=  0;
  nbOutputLastSumOfSizes_  =  0;
  
}


//______________________________________________________________________________
double FUResourceBroker::deltaT(const struct timeval *start,
				const struct timeval *end)
{
  unsigned int  sec;
  unsigned int  usec;
  
  sec = end->tv_sec - start->tv_sec;
  
  if(end->tv_usec > start->tv_usec) {
    usec = end->tv_usec - start->tv_usec;
  }
  else {
    sec--;
    usec = 1000000 - ((unsigned int )(start->tv_usec - end->tv_usec));
  }
  
  return ((double)sec) + ((double)usec) / 1000000.0;
}


////////////////////////////////////////////////////////////////////////////////
// XDAQ instantiator implementation macro
////////////////////////////////////////////////////////////////////////////////

XDAQ_INSTANTIATOR_IMPL(FUResourceBroker)
