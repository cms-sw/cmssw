////////////////////////////////////////////////////////////////////////////////
//
// FUResourceBroker
// ----------------
//
//            10/20/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ResourceBroker/interface/FUResourceBroker.h"
#include "EventFilter/ResourceBroker/interface/FUResource.h"
#include "EventFilter/ResourceBroker/interface/BUProxy.h"
#include "EventFilter/ResourceBroker/interface/SMProxy.h"

#include "FWCore/Utilities/interface/CRC16.h"

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

#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/HTMLClasses.h"

#include <signal.h>
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
  , wlWatching_(0)
  , asWatching_(0)
  , instance_(0)
  , runNumber_(0)
  , deltaT_(0.0)
  , deltaN_(0)
  , deltaSumOfSquares_(0)
  , deltaSumOfSizes_(0)
  , throughput_(0.0)
  , rate_(0.0)
  , average_(0.0)
  , rms_(0.0)
  , nbAllocatedEvents_(0)
  , nbPendingRequests_(0)
  , nbReceivedEvents_(0)
  , nbSentEvents_(0)
  , nbSentErrorEvents_(0)
  , nbPendingSMDiscards_(0)
  , nbDiscardedEvents_(0)
  , nbLostEvents_(0)
  , nbDataErrors_(0)
  , nbCrcErrors_(0)
  , segmentationMode_(false)
  , nbClients_(0)
  , clientPrcIds_("")
  , nbRawCells_(16)
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
  , monSleepSec_(2)
  , watchSleepSec_(10)
  , timeOutSec_(30)
  , processKillerEnabled_(true)
  , useEvmBoard_(true)
  , reasonForFailed_("")
  , nbAllocateSent_(0)
  , nbTakeReceived_(0)
  , nbDataDiscardReceived_(0)
  , nbDqmDiscardReceived_(0)
  , nbSentLast_(0)
  , sumOfSquaresLast_(0)
  , sumOfSizesLast_(0)
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
  xgi::bind(this,&evf::FUResourceBroker::customWebPage,"customWebPage");
  

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

  // set application icon for hyperdaq
  getApplicationDescriptor()->setAttribute("icon", "/evf/images/rbicon.jpg");
  FUResource::useEvmBoard_ = useEvmBoard_;
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
    FUResource::doFedIdCheck(doFedIdCheck_);
    resourceTable_->setDoCrcCheck(doCrcCheck_);
    resourceTable_->setDoDumpEvents(doDumpEvents_);
    reset();
    LOG4CPLUS_INFO(log_, "Finished configuring!");
    
    fsm_.fireEvent("ConfigureDone",this);
  }
  catch (xcept::Exception &e) {
    std::string msg  = "configuring FAILED: " + (string)e.what();
    reasonForFailed_ = e.what();
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
    startWatchingWorkLoop();
    resourceTable_->resetCounters();
    resourceTable_->startDiscardWorkLoop();
    resourceTable_->startSendDataWorkLoop();
    resourceTable_->startSendDqmWorkLoop();
    resourceTable_->sendAllocate();
    LOG4CPLUS_INFO(log_, "Finished enabling!");
    fsm_.fireEvent("EnableDone",this);
  }
  catch (xcept::Exception &e) {
    std::string msg  = "enabling FAILED: "+xcept::stdformat_exception_history(e);
    reasonForFailed_ = e.what();
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
	count++;
	LOG4CPLUS_DEBUG(log_,"Waiting for ResourceTable to shutdown ("<<count<<")");
	::sleep(1);
      }
    }
    
    if (count<10) {
      LOG4CPLUS_INFO(log_, "Finished stopping!");
      fsm_.fireEvent("StopDone",this);
    }
    else {
      std::string msg  = "stopping FAILED: ResourceTable shutdown timed out.";
      reasonForFailed_ = "RESOURCETABLE SHUTDOWN TIMED OUT.";
      fsm_.fireFailed(msg,this);
    }
  }
  catch (xcept::Exception &e) {
    std::string msg  = "stopping FAILED: "+xcept::stdformat_exception_history(e);
    reasonForFailed_ = e.what();
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
	  count++;
	  LOG4CPLUS_DEBUG(log_,count<<". try to destroy resource table failed ...");
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
      std::string msg  = "halting FAILED: ResourceTable shutdown timed out.";
      reasonForFailed_ = "RESOURCETABLE SHUTDOWN TIMED OUT";
      fsm_.fireFailed(msg,this);
    }
  }
  catch (xcept::Exception &e) {
    std::string msg  = "halting FAILED: "+xcept::stdformat_exception_history(e);
    reasonForFailed_ = e.what();
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
    nbClients_          =resourceTable_->nbClients();
    clientPrcIds_       =resourceTable_->clientPrcIdsAsString();
    nbAllocatedEvents_  =resourceTable_->nbAllocated();
    nbPendingRequests_  =resourceTable_->nbPending();
    nbReceivedEvents_   =resourceTable_->nbCompleted();
    nbSentEvents_       =resourceTable_->nbSent();
    nbSentErrorEvents_  =resourceTable_->nbSentError();
    nbPendingSMDiscards_=resourceTable_->nbPendingSMDiscards();
    nbDiscardedEvents_  =resourceTable_->nbDiscarded();
    nbLostEvents_       =resourceTable_->nbLost();
    nbDataErrors_       =resourceTable_->nbErrors();
    nbCrcErrors_        =resourceTable_->nbCrcErrors();
    nbAllocateSent_     =resourceTable_->nbAllocSent();
  }
  else if (e.type()=="ItemChangedEvent") {
    
    string item=dynamic_cast<xdata::ItemChangedEvent&>(e).itemName();
    
    if (item=="doFedIdCheck") FUResource::doFedIdCheck(doFedIdCheck_);
    if (item=="doCrcCheck")   resourceTable_->setDoCrcCheck(doCrcCheck_);
    if (item=="doDumpEvents") resourceTable_->setDoDumpEvents(doDumpEvents_);
    //if (item=="runNumber")    resourceTable_->resetCounters();
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
  
  unsigned int nbSent      =resourceTable_->nbSent();
  uint64_t     sumOfSquares=resourceTable_->sumOfSquares();
  unsigned int sumOfSizes  =resourceTable_->sumOfSizes();

  uint64_t     deltaSumOfSquares;
  
  gui_->monInfoSpace()->lock();
  
  deltaT_.value_=deltaT(&monStartTime_,&monEndTime);
  monStartTime_=monEndTime;
  
  deltaN_.value_=nbSent-nbSentLast_;
  nbSentLast_=nbSent;
  
  deltaSumOfSquares=sumOfSquares-sumOfSquaresLast_;
  deltaSumOfSquares_.value_=(double)deltaSumOfSquares;
  sumOfSquaresLast_=sumOfSquares;
  
  deltaSumOfSizes_.value_=sumOfSizes-sumOfSizesLast_;
  sumOfSizesLast_=sumOfSizes;
  
  if (deltaT_.value_!=0) {
    throughput_=deltaSumOfSizes_.value_/deltaT_.value_;
    rate_      =deltaN_.value_/deltaT_.value_;
  }
  else {
    throughput_=0.0;
    rate_      =0.0;
  }
  
  double meanOfSquares,mean,squareOfMean,variance;
  
  if(deltaN_.value_!=0) {
    meanOfSquares=deltaSumOfSquares_.value_/((double)(deltaN_.value_));
    mean=((double)(deltaSumOfSizes_.value_))/((double)(deltaN_.value_));
    squareOfMean=mean*mean;
    variance=meanOfSquares-squareOfMean; if(variance<0.0) variance=0.0;

    average_=deltaSumOfSizes_.value_/deltaN_.value_;
    rms_    =std::sqrt(variance);
  }
  else {
    average_=0.0;
    rms_    =0.0;
  }
  
  gui_->monInfoSpace()->unlock();  
  
  ::sleep(monSleepSec_.value_);
  
  return true;
}
    

//______________________________________________________________________________
void FUResourceBroker::startWatchingWorkLoop() throw (evf::Exception)
{
  try {
    wlWatching_=
      toolbox::task::getWorkLoopFactory()->getWorkLoop(sourceId_+"Watching",
						       "waiting");
    if (!wlWatching_->isActive()) wlWatching_->activate();
    asWatching_=toolbox::task::bind(this,&FUResourceBroker::watching,
				    sourceId_+"Watching");
    wlWatching_->submit(asWatching_);
  }
  catch (xcept::Exception& e) {
    string msg = "Failed to start workloop 'Watching'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
}


//______________________________________________________________________________
bool FUResourceBroker::watching(toolbox::task::WorkLoop* wl)
{
  if (0==resourceTable_) return false;
  
  vector<pid_t> prcids=resourceTable_->clientPrcIds();
  for (UInt_t i=0;i<prcids.size();i++) {
    pid_t pid   =prcids[i];
    int   status=kill(pid,0);
    if (status!=0) {
      LOG4CPLUS_ERROR(log_,"EP prc "<<pid<<" died, send raw data to err stream.");
      resourceTable_->handleCrashedEP(runNumber_,pid);
    }
  }
  
  resourceTable_->lockShm();
  vector<pid_t>  evt_prcids =resourceTable_->cellPrcIds();
  vector<UInt_t> evt_numbers=resourceTable_->cellEvtNumbers();
  vector<time_t> evt_tstamps=resourceTable_->cellTimeStamps(); 
  resourceTable_->unlockShm();

  time_t tcurr=time(0);  
  for (UInt_t i=0;i<evt_tstamps.size();i++) {
    pid_t  pid   =evt_prcids[i];
    UInt_t evt   =evt_numbers[i];
    time_t tstamp=evt_tstamps[i]; if (tstamp==0) continue;
    double tdiff =difftime(tcurr,tstamp);
    if (tdiff>timeOutSec_) {
      if(processKillerEnabled_)	{
	LOG4CPLUS_ERROR(log_,"evt "<<evt<<" timed out, "<<"kill prc "<<pid);
	kill(pid,9);
      }
      else {
	LOG4CPLUS_INFO(log_,"evt "<<evt<<" under processing for more than "
		       <<timeOutSec_<<"sec for process "<<pid);
      }
    }
  }
  
  ::sleep(watchSleepSec_.value_);
  
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

  gui_->addMonitorParam("deltaT",                   &deltaT_);
  gui_->addMonitorParam("deltaN",                   &deltaN_);
  gui_->addMonitorParam("deltaSumOfSquares",        &deltaSumOfSquares_);
  gui_->addMonitorParam("deltaSumOfSizes",          &deltaSumOfSizes_);
    
  gui_->addMonitorParam("throughput",               &throughput_);
  gui_->addMonitorParam("rate",                     &rate_);
  gui_->addMonitorParam("average",                  &average_);
  gui_->addMonitorParam("rms",                      &rms_);
  
  gui_->addMonitorCounter("nbAllocatedEvents",      &nbAllocatedEvents_);
  gui_->addMonitorCounter("nbPendingRequests",      &nbPendingRequests_);
  gui_->addMonitorCounter("nbReceivedEvents",       &nbReceivedEvents_);
  gui_->addMonitorCounter("nbSentEvents",           &nbSentEvents_);
  gui_->addMonitorCounter("nbSentErrorEvents",      &nbSentErrorEvents_);
  gui_->addMonitorCounter("nbPendingSMDiscards",    &nbPendingSMDiscards_);
  gui_->addMonitorCounter("nbDiscardedEvents",      &nbDiscardedEvents_);
  gui_->addMonitorCounter("nbLostEvents",           &nbLostEvents_);
  gui_->addMonitorCounter("nbDataErrors",           &nbDataErrors_);
  gui_->addMonitorCounter("nbCrcErrors",            &nbCrcErrors_);

  gui_->addStandardParam("segmentationMode",        &segmentationMode_);
  gui_->addStandardParam("nbClients",               &nbClients_);
  gui_->addStandardParam("clientPrcIds",            &clientPrcIds_);
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
  gui_->addStandardParam("watchSleepSec",           &watchSleepSec_);
  gui_->addStandardParam("timeOutSec",              &timeOutSec_);
  gui_->addStandardParam("processKillerEnabled",    &processKillerEnabled_);
  gui_->addStandardParam("useEvmBoard",             &useEvmBoard_);
  gui_->addStandardParam("foundRcmsStateListener",   fsm_.foundRcmsStateListener());
  gui_->addStandardParam("reasonForFailed",         &reasonForFailed_);
  
  gui_->addDebugCounter("nbAllocateSent",           &nbAllocateSent_);
  gui_->addDebugCounter("nbTakeReceived",           &nbTakeReceived_);
  gui_->addDebugCounter("nbDataDiscardReceived",    &nbDataDiscardReceived_);
  gui_->addDebugCounter("nbDqmDiscardReceived",     &nbDqmDiscardReceived_);

  gui_->exportParameters();

  gui_->addItemChangedListener("doFedIdCheck",      this);
  gui_->addItemChangedListener("doCrcCheck",        this);
  gui_->addItemChangedListener("doDumpEvents",      this);
  //gui_->addItemChangedListener("runNumber",         this);
}


//______________________________________________________________________________
void FUResourceBroker::reset()
{
  gui_->resetCounters();
  
  deltaT_           =0.0;
  deltaN_           =  0;
  deltaSumOfSquares_=0.0;
  deltaSumOfSizes_  =  0;
  
  throughput_       =0.0;
  rate_             =0.0;
  average_          =0.0;
  rms_              =0.0;
  
  nbSentLast_       =  0;
  sumOfSquaresLast_ =  0;
  sumOfSizesLast_   =  0;
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



//______________________________________________________________________________
void FUResourceBroker::customWebPage(xgi::Input*in,xgi::Output*out)
  throw (xgi::exception::Exception)
{
  using namespace cgicc;
  
  *out<<"<html>"<<endl;
  gui_->htmlHead(in,out,sourceId_);
  *out<<"<body>"<<endl;
  gui_->htmlHeadline(in,out);

  if (0!=resourceTable_) {
    vector<pid_t> client_prc_ids = resourceTable_->clientPrcIds();
    *out<<table().set("frame","void").set("rules","rows")
                 .set("class","modules").set("width","250")<<endl
	<<tr()<<th("Client Processes").set("colspan","3")<<tr()<<endl
	<<tr()
	<<th("client").set("align","left")
	<<th("process id").set("align","center")
	<<th("status").set("align","center")
	<<tr()
	<<endl;
    for (UInt_t i=0;i<client_prc_ids.size();i++) {

      pid_t pid   =client_prc_ids[i];
      int   status=kill(pid,0);

      stringstream ssi;      ssi<<i+1;
      stringstream sspid;    sspid<<pid;
      stringstream ssstatus; ssstatus<<status;
      
      string bg_status = (status==0) ? "#00ff00" : "ff0000";
      *out<<tr()
	  <<td(ssi.str()).set("align","left")
	  <<td(sspid.str()).set("align","center")
	  <<td(ssstatus.str()).set("align","center").set("bgcolor",bg_status)
	  <<tr()<<endl;
    }
    *out<<table()<<endl;
    *out<<"<br><br>"<<endl;

    resourceTable_->lockShm();
    vector<string> states      = resourceTable_->cellStates();
    vector<UInt_t> evt_numbers = resourceTable_->cellEvtNumbers();
    vector<pid_t>  prc_ids     = resourceTable_->cellPrcIds();
    vector<time_t> time_stamps = resourceTable_->cellTimeStamps();
    resourceTable_->unlockShm();

    *out<<table().set("frame","void").set("rules","rows")
                 .set("class","modules").set("width","500")<<endl
	<<tr()<<th("Shared Memory Cells").set("colspan","6")<<tr()<<endl
	<<tr()
	<<th("cell").set("align","left")
	<<th("state").set("align","center")
	<<th("event").set("align","center")
	<<th("process id").set("align","center")
	<<th("timestamp").set("align","center")
	<<th("time").set("align","center")
	<<tr()
	<<endl;
    for (UInt_t i=0;i<states.size();i++) {
      string state=states[i];
      UInt_t evt   = evt_numbers[i];
      pid_t  pid   = prc_ids[i];
      time_t tstamp= time_stamps[i];
      double tdiff = difftime(time(0),tstamp);
      
      stringstream ssi;      ssi<<i;
      stringstream ssevt;    if (evt!=0xffffffff) ssevt<<evt; else ssevt<<" - ";
      stringstream sspid;    if (pid!=0) sspid<<pid; else sspid<<" - ";
      stringstream sststamp; if (tstamp!=0) sststamp<<tstamp; else sststamp<<" - ";
      stringstream sstdiff;  if (tstamp!=0) sstdiff<<tdiff; else sstdiff<<" - ";
      
      string bg_state = "#ffffff";
      if (state=="RAWWRITING"||state=="RAWWRITTEN"||
	  state=="RAWREADING"||state=="RAWREAD")
	bg_state="#99CCff";
      else if (state=="PROCESSING")
	bg_state="#ff0000";
      else if (state=="PROCESSED"||state=="RECOWRITING"||state=="RECOWRITTEN")
	bg_state="#CCff99";
      else if (state=="SENDING")
	bg_state="#00FF33";
      else if (state=="SENT")
	bg_state="#006633";
      else if (state=="DISCARDING")
	bg_state="#FFFF00";
      
      *out<<tr()
	  <<td(ssi.str()).set("align","left")
	  <<td(state).set("align","center").set("bgcolor",bg_state)
	  <<td(ssevt.str()).set("align","center")
	  <<td(sspid.str()).set("align","center")
	  <<td(sststamp.str()).set("align","center")
	  <<td(sstdiff.str()).set("align","center")
	  <<tr()<<endl;
    }
    *out<<table()<<endl;

    
  }
  *out<<"</body>"<<endl<<"</html>"<<endl;
}



////////////////////////////////////////////////////////////////////////////////
// XDAQ instantiator implementation macro
////////////////////////////////////////////////////////////////////////////////

XDAQ_INSTANTIATOR_IMPL(FUResourceBroker)
