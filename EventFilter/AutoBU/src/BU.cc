////////////////////////////////////////////////////////////////////////////////
//
// BU
// --
//
//                                         Emilio Meschi <emilio.meschi@cern.ch>
//                       Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/AutoBU/interface/BU.h"

#include "EventFilter/Utilities/interface/Crc.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "xoap/include/xoap/SOAPEnvelope.h"
#include "xoap/include/xoap/SOAPBody.h"
#include "xoap/include/xoap/domutils.h"

#include <netinet/in.h>

#include <sstream>


using namespace std;
using namespace evf;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
BU::BU(xdaq::ApplicationStub *s) 
  : xdaq::Application(s)
  , log_(getApplicationLogger())
  , fsm_(0)
  , gui_(0)
  , fedN_(0)
  , fedData_(0)
  , fedSize_(0)
  , fedId_(0)
  , instance_(0)
  , runNumber_(0)
  , nbMBTot_(0.0)
  , nbMBPerSec_(0.0)
  , nbMBPerSecMin_(0.0)
  , nbMBPerSecMax_(0.0)
  , nbMBPerSecAvg_(0.0)
  , memUsedInMB_(0.0)
  , nbEventsInBU_(0)
  , deltaT_(0.0)
  , deltaN_(0)
  , deltaSumOfSquares_(0)
  , deltaSumOfSizes_(0)
  , nbEvents_(0)
  , nbEventsPerSec_(0)
  , nbEventsPerSecMin_(0)
  , nbEventsPerSecMax_(0)
  , nbEventsPerSecAvg_(0)
  , nbDiscardedEvents_(0)
  , mode_("RANDOM")
  , dataBufSize_(32768)
  , nSuperFrag_(64)
  , fedSizeMax_(65536)
  , fedSizeMean_(1024)   // mean  of fed size for rnd generation
  , fedSizeWidth_(256)   // width of fed size for rnd generation
  , useFixedFedSize_(false)
  , nbPostFrame_(0)
  , nbPostFrameFailed_(0)
  , nbMeasurements_(0)
  , nbEventsLast_(0)
  , nbBytes_(0)
  , sumOfSquares_(0)
  , sumOfSizes_(0)
  , i2oPool_(0)
  , lock_(BSem::FULL)
{
  // initialize application info
  url_     =
    getApplicationDescriptor()->getContextDescriptor()->getURL()+"/"+
    getApplicationDescriptor()->getURN();
  class_   =getApplicationDescriptor()->getClassName();
  instance_=getApplicationDescriptor()->getInstance();
  hostname_=getApplicationDescriptor()->getContextDescriptor()->getURL();
  sourceId_=class_.toString()+instance_.toString();
  
  // initialize state machine
  fsm_=new EPStateMachine(log_);
  fsm_->init<BU>(this);
  
  // web interface
  xgi::bind(this,&evf::BU::webPageRequest,"Default");
  gui_=new WebGUI(this,fsm_);
  gui_->setSmallAppIcon("/daq/evb/bu/images/bu32x32.gif");
  gui_->setLargeAppIcon("/daq/evb/bu/images/bu64x64.gif");
  
  vector<toolbox::lang::Method*> methods=gui_->getMethods();
  vector<toolbox::lang::Method*>::iterator it;
  for (it=methods.begin();it!=methods.end();++it) {
    if ((*it)->type()=="cgi") {
      string name=static_cast<xgi::MethodSignature*>(*it)->name();
      xgi::bind(this,&evf::BU::webPageRequest,name);
    }
  }
  
  // i2o callbacks
  i2o::bind(this,&BU::I2O_BU_ALLOCATE_Callback,I2O_BU_ALLOCATE,XDAQ_ORGANIZATION_ID);
  i2o::bind(this,&BU::I2O_BU_COLLECT_Callback, I2O_BU_COLLECT, XDAQ_ORGANIZATION_ID);
  i2o::bind(this,&BU::I2O_BU_DISCARD_Callback, I2O_BU_DISCARD, XDAQ_ORGANIZATION_ID);
  
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
  
  // export parameters to info space(s)
  exportParameters();
}


//______________________________________________________________________________
BU::~BU()
{
  clearFedBuffers();
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void BU::timeExpired(toolbox::task::TimerEvent& e)
{
  
  lock();
  gui_->lockInfoSpaces();

  nbMeasurements_++;
  
  // number of events per second measurement
  nbEventsPerSec_=nbEvents_-nbEventsLast_;
  nbEventsLast_  =nbEvents_;
  if (nbEventsPerSec_.value_>0) {
    if (nbEventsPerSec_<nbEventsPerSecMin_) nbEventsPerSecMin_=nbEventsPerSec_;
    if (nbEventsPerSec_>nbEventsPerSecMax_) nbEventsPerSecMax_=nbEventsPerSec_;
  }
  nbEventsPerSecAvg_=nbEvents_/nbMeasurements_;

  // throughput measurements
  nbMBPerSec_=9.53674e-07*nbBytes_;
  nbBytes_=0;
  nbMBTot_.value_+=nbMBPerSec_;
  if (nbMBPerSec_.value_>0.0) {
    if (nbMBPerSec_<nbMBPerSecMin_) nbMBPerSecMin_=nbMBPerSec_;
    if (nbMBPerSec_>nbMBPerSecMax_) nbMBPerSecMax_=nbMBPerSec_;
  }
  nbMBPerSecAvg_=nbMBTot_/nbMeasurements_;
  
  // time interval since last evaluation
  deltaT_=1.0;

  // number of events processed in that interval
  deltaN_=nbEventsPerSec_;
  
  // sum of squared events sizes
  deltaSumOfSquares_=sumOfSquares_;
  sumOfSquares_=0;

  // sum of squared events sizes
  deltaSumOfSizes_=sumOfSizes_;
  sumOfSizes_=0;

  gui_->unlockInfoSpaces();
  unlock();
}


//______________________________________________________________________________
void BU::initTimer()
{
  toolbox::task::getTimerFactory()->createTimer(sourceId_);
  toolbox::task::Timer *timer=toolbox::task::getTimerFactory()->getTimer(sourceId_);
  timer->stop();
}


//______________________________________________________________________________
void BU::startTimer()
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
    timer->scheduleAtFixedRate(startTime,this,oneSec,gui_->appInfoSpace(),sourceId_);
  }
}


//______________________________________________________________________________
void BU::stopTimer()
{
  toolbox::task::Timer *timer =toolbox::task::getTimerFactory()->getTimer(sourceId_);
  timer->stop();
  toolbox::task::getTimerFactory()->removeTimer(sourceId_);
}


//______________________________________________________________________________
void BU::actionPerformed(xdata::Event& e)
{
  gui_->lockInfoSpaces();
  if (e.type()=="ItemRetrieveEvent") {
    string item=dynamic_cast<xdata::ItemRetrieveEvent&>(e).itemName();
    if (item=="mode") {
      mode_=(0==PlaybackRawDataProvider::instance())?"RANDOM":"PLAYBACK";
    }
    if (item=="memUsedInMB") {
      if (0!=i2oPool_) memUsedInMB_=i2oPool_->getMemoryUsage().getUsed()*0.000001;
      else             memUsedInMB_=0.0;
    }
  }
  gui_->unlockInfoSpaces();
}


//______________________________________________________________________________
void BU::configureAction(toolbox::Event::Reference e) 
  throw (toolbox::fsm::exception::Exception)
{
  // reset all relevant variables to 'configured' state
  reset();

  // intialize timer for performance measurements
  initTimer();
  
  LOG4CPLUS_INFO(log_,"BU -> CONFIGURED <-");
}


//______________________________________________________________________________
void BU::enableAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  LOG4CPLUS_INFO(log_,"BU -> ENABLED <-");
}


//______________________________________________________________________________
void BU::suspendAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  lock();
  LOG4CPLUS_INFO(log_,"BU -> SUSPENDED <-");
}


//______________________________________________________________________________
void BU::resumeAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  unlock();
  LOG4CPLUS_INFO(log_,"BU -> RESUMED <-");
}


//______________________________________________________________________________
void BU::haltAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  // stop timer
  stopTimer();

  LOG4CPLUS_INFO(log_,"BU -> HALTED <-");
}


//______________________________________________________________________________
void BU::nullAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  LOG4CPLUS_INFO(log_,"BU::nullAction()");
}


//______________________________________________________________________________
xoap::MessageReference BU::fireEvent(xoap::MessageReference msg)
  throw (xoap::exception::Exception)
{
  xoap::SOAPPart     part    =msg->getSOAPPart();
  xoap::SOAPEnvelope env     =part.getEnvelope();
  xoap::SOAPBody     body    =env.getBody();
  DOMNode           *node    =body.getDOMNode();
  DOMNodeList       *bodyList=node->getChildNodes();
  DOMNode           *command =0;
  string             commandName;
  
  for (unsigned int i=0;i<bodyList->getLength();i++) {
    command = bodyList->item(i);
    if(command->getNodeType() == DOMNode::ELEMENT_NODE) {
      commandName = xoap::XMLCh2String(command->getLocalName());
      return fsm_->processFSMCommand(commandName);
    }
  }
  XCEPT_RAISE(xoap::exception::Exception,"Command not found");
}


//______________________________________________________________________________
void BU::webPageRequest(xgi::Input *in,xgi::Output *out)
  throw (xgi::exception::Exception)
{
  string name=in->getenv("PATH_INFO");
  if (name.empty()) name="defaultWebPage";
  static_cast<xgi::MethodSignature*>(gui_->getMethod(name))->invoke(in,out);
}


//______________________________________________________________________________
void BU::I2O_BU_ALLOCATE_Callback(toolbox::mem::Reference *bufRef)
{
  // check if the BU is enabled
  toolbox::fsm::State currentState=fsm_->getCurrentState();
  if (currentState!='E') {
    LOG4CPLUS_WARN(log_,"Ignore I2O_BU_ALLOCATE while *not* enabled!");
    bufRef->release();
    return;
  }

  // start timer for performance measurements upon receiving first request
  if (nbEvents_.value_==0) startTimer();
  
  
  // process message
  I2O_MESSAGE_FRAME             *stdMsg;
  I2O_BU_ALLOCATE_MESSAGE_FRAME *msg;

  stdMsg=(I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
  msg   =(I2O_BU_ALLOCATE_MESSAGE_FRAME*)stdMsg;

  // loop over all requested events
  
  unsigned int nbEvents=msg->n; nbEventsInBU_=nbEvents;
  for(unsigned int i=0;i<nbEvents;i++) {
    
    unsigned int evtSizeInBytes(0);

    U32     fuTransactionId=msg   ->allocate[i].fuTransactionId; // assigned by FU
    I2O_TID fuTid          =stdMsg->InitiatorAddress;            // declared to i2o
    
    // if a raw data provider is present, request an event from it
    unsigned int runNumber=0; // not needed
    unsigned int evtNumber=(nbEvents_+1)%0x1000000;
    FEDRawDataCollection* event(0);
    if (0!=PlaybackRawDataProvider::instance())
      event=PlaybackRawDataProvider::instance()->getFEDRawData(runNumber,evtNumber);
    
    
    //
    // loop over all superfragments in each event    
    //
    for (unsigned int iSuperFrag=0;iSuperFrag<nSuperFrag_;iSuperFrag++) {
      
      // fill FED buffers
      fillFedBuffers(iSuperFrag,event);

      // create super fragment
      toolbox::mem::Reference *superFrag=
	createSuperFrag(fuTid,           // fuTid
			fuTransactionId, // fuTransaction
			evtNumber,       // current trigger (event) number
			iSuperFrag,      // current super fragment
			nSuperFrag_      // number of super fragments
			);
      
      I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *frame =
	(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)(superFrag->getDataLocation());
      
      unsigned int msgSizeInBytes=
	frame->PvtMessageFrame.StdMessageFrame.MessageSize<<2;

      superFrag->setDataSize(msgSizeInBytes);
      lock();
      nbBytes_      +=msgSizeInBytes;
      evtSizeInBytes+=msgSizeInBytes;
      unlock();
      
      xdaq::ApplicationDescriptor *buAppDesc=
	getApplicationDescriptor();
      
      xdaq::ApplicationDescriptor *fuAppDesc= 
	i2o::utils::getAddressMap()->getApplicationDescriptor(fuTid);
      
      // post frame until it succeeds
      bool framePosted(false);
      while (!framePosted) {
	try {
	  getApplicationContext()->postFrame(superFrag,buAppDesc,fuAppDesc);
	  nbPostFrame_.value_++;
	  framePosted=true;
	}
	catch (xcept::Exception& e) {
	  nbPostFrameFailed_.value_++;
	}
      }
      
    }
    
    if (0!=event) delete event;
    
    lock();
    nbEvents_.value_++;
    nbEventsInBU_.value_--;
    sumOfSquares_+=evtSizeInBytes*evtSizeInBytes;
    sumOfSizes_  +=evtSizeInBytes;
    unlock();
  }

  // Free the request message from the FU
  bufRef->release();
}


//______________________________________________________________________________
void BU::I2O_BU_COLLECT_Callback(toolbox::mem::Reference *bufRef)
{
  bufRef->release();
  LOG4CPLUS_WARN(log_,"I2O_BU_COLLECT_Callback() not implemented!");
}


//______________________________________________________________________________
void BU::I2O_BU_DISCARD_Callback(toolbox::mem::Reference *bufRef)
{
  bufRef->release();
  nbDiscardedEvents_.value_++;
}


//______________________________________________________________________________
void BU::initFedBuffers(unsigned int nFed)
{
  clearFedBuffers();

  if (nFed<nSuperFrag_) nSuperFrag_=nFed;
  
  assert(nSuperFrag_.value_>0);
  
  fedN_ =new unsigned int[nSuperFrag_];
  fedId_=new unsigned int*[nSuperFrag_];
  
  unsigned int fedN1=nFed/nSuperFrag_;
  unsigned int fedN2=fedN1;
  unsigned int mod  =nFed%nSuperFrag_;
  if (mod>0) fedN2++;
  
  for (unsigned int i=0;i<nSuperFrag_;i++) {
    fedN_[i] =(i<mod)?fedN2:fedN1;
    fedId_[i]=new unsigned int[fedN_[i]];
  }

  fedData_=new unsigned char*[fedN2];
  fedSize_=new unsigned int[fedN2];

  for (unsigned int i=0;i<fedN2;i++) {    
    fedData_[i]=new unsigned char[fedSizeMax_];
    fedSize_[i]=0;
  }

}


//______________________________________________________________________________
void BU::clearFedBuffers()
{
  if (0!=fedData_) {
    for (unsigned int i=0;i<fedN_[0];i++) delete [] fedData_[i];
    delete [] fedData_;
    fedData_=0;
  }

  if (0!=fedSize_) {
    delete [] fedSize_;
    fedSize_=0;
  }
  
  if (0!=fedId_) {
    for (unsigned int i=0;i<nSuperFrag_;i++) delete [] fedId_[i];
    delete [] fedId_;
    fedId_=0;
  }

  if (0!=fedN_) delete [] fedN_; fedN_=0;
}


//______________________________________________________________________________
void BU::fillFedBuffers(unsigned int iSuperFrag,FEDRawDataCollection* event)
{
  // determine valid FED Ids and number of FEDs per superfragment if necessary
  if (fedN_==0) {
    vector<unsigned int> validFedIds;
    for (unsigned int i=0;i<(unsigned int)FEDNumbering::lastFEDId()+1;i++) {
      if (0==event) {
	if (FEDNumbering::inRange(i)) validFedIds.push_back(i);
      }
      else {
	unsigned int fedSize=event->FEDData(i).size();
	if (fedSize>0) validFedIds.push_back(i);
      }
    }
    unsigned int nFed=validFedIds.size();
    assert(nFed>0);
    
    initFedBuffers(nFed);
    
    unsigned int i(0),j(0);
    for (unsigned int k=0;k<nFed;k++) {
      unsigned int fedId=validFedIds[k];
      fedId_[i][j]=fedId;
      j++;
      if (j%fedN_[i]==0) { j=0; i++; }
    }
  }
  
  // generate random FEDs (RANDOM mode)
  if (0==event) {
    if(useFixedFedSize_) {
      for (unsigned int i=0;i<fedN_[iSuperFrag];i++) fedSize_[i]=fedSizeMean_;
    }
    else {
      for(unsigned int i=0;i<fedN_[iSuperFrag];i++) {
	unsigned int iFedSize(0);
	while (iFedSize<(fedTrailerSize_+fedHeaderSize_)||iFedSize>fedSizeMax_) {
	  double logSize=RandGauss::shoot(std::log((double)fedSizeMean_),
					  std::log((double)fedSizeMean_)-
					  std::log((double)fedSizeWidth_/2.));
	  iFedSize=(int)(std::exp(logSize));
	  iFedSize-=iFedSize % 8; // all blocks aligned to 64 bit words
	}
	fedSize_[i]=iFedSize;
      }
    }
  }
  // generate FEDs from data file (PLAYBACK mode)
  else {
    for (unsigned int i=0;i<fedN_[iSuperFrag];i++) {
      fedSize_[i]=event->FEDData(fedId_[iSuperFrag][i]).size();
      fedData_[i]=event->FEDData(fedId_[iSuperFrag][i]).data();
    }
  }
  
}


//______________________________________________________________________________
void BU::exportParameters()
{
  if (0==gui_) {
    LOG4CPLUS_ERROR(log_,"No GUI, can't export parameters");
    return;
  }
  
  gui_->addMonitorParam("url",                &url_);
  gui_->addMonitorParam("class",              &class_);
  gui_->addMonitorParam("instance",           &instance_);
  gui_->addMonitorParam("hostname",           &hostname_);
  gui_->addMonitorParam("runNumber",          &runNumber_);
  gui_->addMonitorParam("stateName",          &fsm_->stateName_);
  gui_->addMonitorParam("nbMBTot",            &nbMBTot_);
  gui_->addMonitorParam("nbMBPerSec",         &nbMBPerSec_);
  gui_->addMonitorParam("nbMBPerSecMin",      &nbMBPerSecMin_);
  gui_->addMonitorParam("nbMBPerSecMax",      &nbMBPerSecMax_);
  gui_->addMonitorParam("nbMBPerSecAvg",      &nbMBPerSecAvg_);
  gui_->addMonitorParam("memUsedInMB",        &memUsedInMB_);
  gui_->addMonitorParam("nbEventsInBU",       &nbEventsInBU_);
  gui_->addMonitorParam("deltaT",             &deltaT_);
  gui_->addMonitorParam("deltaN",             &deltaN_);
  gui_->addMonitorParam("deltaSumOfSquares",  &deltaSumOfSquares_);
  gui_->addMonitorParam("deltaSumOfSizes",    &deltaSumOfSizes_);

  
  gui_->addMonitorCounter("nbEvents",         &nbEvents_);
  gui_->addMonitorCounter("nbEventsPerSec",   &nbEventsPerSec_);
  gui_->addMonitorCounter("nbEventsPerSecMin",&nbEventsPerSecMin_);
  gui_->addMonitorCounter("nbEventsPerSecMax",&nbEventsPerSecMax_);
  gui_->addMonitorCounter("nbEventsPerSecAvg",&nbEventsPerSecAvg_);
  gui_->addMonitorCounter("nbDiscardedEvents",&nbDiscardedEvents_);

  gui_->addStandardParam("mode",              &mode_);
  gui_->addStandardParam("dataBufSize",       &dataBufSize_);
  gui_->addStandardParam("nSuperFrag",        &nSuperFrag_);
  gui_->addStandardParam("fedSizeMax",        &fedSizeMax_);
  gui_->addStandardParam("fedSizeMean",       &fedSizeMean_);
  gui_->addStandardParam("fedSizeWidth",      &fedSizeWidth_);
  gui_->addStandardParam("useFixedFedSize",   &useFixedFedSize_);
  
  gui_->addDebugCounter("nbPostFrame",        &nbPostFrame_);
  gui_->addDebugCounter("nbPostFrameFailed",  &nbPostFrameFailed_);
  
  gui_->exportParameters();

  gui_->addItemRetrieveListener("mode",       this);
  gui_->addItemRetrieveListener("memUsedInMB",this);
}


//______________________________________________________________________________
void BU::reset()
{
  gui_->resetCounters();
  
  nbEventsPerSecMin_=10000;
  nbMBTot_          =0.0;
  nbMBPerSec_       =0.0;
  nbMBPerSecMin_    =1e6;
  nbMBPerSecMax_    =0.0;
  nbMBPerSecAvg_    =0.0;
  nbEventsInBU_     =0;
  deltaN_           =0;
  deltaSumOfSquares_=0;
  deltaSumOfSizes_  =0;
  
  nbMeasurements_   =0;
  nbEventsLast_     =0;
  nbBytes_          =0;
  sumOfSquares_     =0;
  sumOfSizes_       =0;
}
  

////////////////////////////////////////////////////////////////////////////////
// implementation of private member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
toolbox::mem::Reference *BU::createSuperFrag(const I2O_TID& fuTid,
					     const U32&     fuTransaction,
					     const U32&     trigNo,
					     const U32&     iSuperFrag,
					     const U32&     nSuperFrag)
{
  bool         configFeds      =(0==PlaybackRawDataProvider::instance());
  unsigned int msgHeaderSize   =sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
  unsigned int fullBlockPayload=dataBufSize_-msgHeaderSize;
  
  if((fullBlockPayload%4)!=0)
    LOG4CPLUS_ERROR(log_,"The full block payload of "<<fullBlockPayload
		    <<" bytes is not a multiple of 4");
  
  unsigned int nBlock=estimateNBlocks(iSuperFrag,fullBlockPayload);
  
  toolbox::mem::Reference *head  =0;
  toolbox::mem::Reference *tail  =0;
  toolbox::mem::Reference *bufRef=0;

  I2O_MESSAGE_FRAME                  *stdMsg=0;
  I2O_PRIVATE_MESSAGE_FRAME          *pvtMsg=0;
  I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block =0;

  
  //
  // loop over all superfragment blocks
  //
  unsigned int   iFed          =0;
  unsigned int   remainder     =0;
  bool           fedTrailerLeft=false;
  bool           last          =false;
  bool           warning       =false;
  unsigned char *startOfPayload=0;
  U32            payload(0);
  
  for(unsigned int iBlock=0;iBlock<nBlock;iBlock++) {
    
    // If last block and its partial (there can be only 0 or 1 partial)
    payload=fullBlockPayload;
    
    // Allocate memory for a fragment block / message
    try	{
      bufRef=toolbox::mem::getMemoryPoolFactory()->getFrame(i2oPool_,dataBufSize_);
    }
    catch(...) {
      LOG4CPLUS_FATAL(log_,"xdaq::frameAlloc failed");
      exit(-1);
    }
    
    // Fill in the fields of the fragment block / message
    stdMsg=(I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
    pvtMsg=(I2O_PRIVATE_MESSAGE_FRAME*)stdMsg;
    block =(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)stdMsg;
    
    memset(block,0,sizeof(I2O_MESSAGE_FRAME));
    
    xdaq::ApplicationDescriptor* buAppDesc=getApplicationDescriptor();
    
    pvtMsg->XFunctionCode   =I2O_FU_TAKE;
    pvtMsg->OrganizationID  =XDAQ_ORGANIZATION_ID;
    
    stdMsg->MessageSize     =(msgHeaderSize + payload) >> 2;
    stdMsg->Function        =I2O_PRIVATE_MESSAGE;
    stdMsg->VersionOffset   =0;
    stdMsg->MsgFlags        =0;  // Point-to-point
    stdMsg->InitiatorAddress=i2o::utils::getAddressMap()->getTid(buAppDesc);
    stdMsg->TargetAddress   =fuTid;

    block->buResourceId           =fuTransaction;
    block->fuTransactionId        =fuTransaction;
    block->blockNb                =iBlock;
    block->nbBlocksInSuperFragment=nBlock;
    block->superFragmentNb        =iSuperFrag;
    block->nbSuperFragmentsInEvent=nSuperFrag;
    block->eventNumber            =trigNo;
    
    // Fill in payload 
    startOfPayload   =(unsigned char*)block+msgHeaderSize;
    frlh_t* frlHeader=(frlh_t*)startOfPayload;
    frlHeader->trigno=trigNo;
    frlHeader->segno =iBlock;
    
    unsigned char *startOfFedBlocks=startOfPayload+frlHeaderSize_;
    payload              -=frlHeaderSize_;
    frlHeader->segsize    =payload;
    unsigned int leftspace=payload;

    // a fed trailer was left over from the previous block
    if(fedTrailerLeft) {
      
      if (configFeds) {
	fedt_t *fedTrailer=(fedt_t*)(fedData_[iFed]+fedSize_[iFed]-fedTrailerSize_);
	fedTrailer->eventsize =fedSize_[iFed];
	fedTrailer->eventsize/=8; //wc in fed trailer in 64bit words
	fedTrailer->eventsize|=0xa0000000;
	fedTrailer->conscheck =0x0;
	unsigned short crc=evf::compute_crc(fedData_[iFed],fedSize_[iFed]);
	fedTrailer->conscheck=(crc<<FED_CRCS_SHIFT);
      }
      
      memcpy(startOfFedBlocks,
	     fedData_[iFed]+fedSize_[iFed]-fedTrailerSize_,fedTrailerSize_);
      
      startOfFedBlocks+=fedTrailerSize_;
      leftspace       -=fedTrailerSize_;
      remainder        =0;
      fedTrailerLeft   =false;
      
      // if this is the last fed, adjust block (msg) size and set last=true
      if((iFed==(fedN_[iSuperFrag]-1)) && !last) {
	frlHeader->segsize-=leftspace;
	int msgSize=stdMsg->MessageSize << 2;
	msgSize   -=leftspace;
	bufRef->setDataSize(msgSize);
	stdMsg->MessageSize = msgSize >> 2;
	frlHeader->segsize=frlHeader->segsize | FRL_LAST_SEGM;
	last=true;
      }
      
      // !! increment iFed !!
      iFed++;
    }
    
    //!
    //! remainder>0 means that a partial fed is left over from the last block
    //!
    if (remainder>0) {
      
      // the remaining fed fits entirely into the new block
      if(payload>=remainder) {

	if (configFeds) {
	  fedt_t *fedTrailer    =(fedt_t*)(fedData_[iFed]+
					   fedSize_[iFed]-fedTrailerSize_);
	  fedTrailer->eventsize =fedSize_[iFed];
	  fedTrailer->eventsize/=8;   //wc in fed trailer in 64bit words
	  fedTrailer->eventsize|=0xa0000000;
	  fedTrailer->conscheck =0x0;
	  unsigned short crc=evf::compute_crc(fedData_[iFed],fedSize_[iFed]);
	  fedTrailer->conscheck=(crc<<FED_CRCS_SHIFT);
	}      
	
	memcpy(startOfFedBlocks,fedData_[iFed]+fedSize_[iFed]-remainder,remainder);
	
	startOfFedBlocks+=remainder;
	leftspace       -=remainder;
	
	// if this is the last fed in the superfragment, earmark it
	if(iFed==fedN_[iSuperFrag]-1) {
	  frlHeader->segsize-=leftspace;
	  int msgSize=stdMsg->MessageSize << 2;
	  msgSize   -=leftspace;
	  bufRef->setDataSize(msgSize);
	  stdMsg->MessageSize = msgSize >> 2;
	  frlHeader->segsize=frlHeader->segsize | FRL_LAST_SEGM;
	  last=true;
	}
	
	// !! increment iFed !!
	iFed++;
	
	// start new fed -> set remainder to 0!
	remainder=0;
      }
      // the remaining payload fits, but not the fed trailer
      else if (payload>=(remainder-fedTrailerSize_)) {
	
	memcpy(startOfFedBlocks,
	       fedData_[iFed]+fedSize_[iFed]-remainder,
	       remainder-fedTrailerSize_);
	
	frlHeader->segsize=remainder-fedTrailerSize_;
	fedTrailerLeft=true;
	leftspace-=(remainder-fedTrailerSize_);
	remainder=fedTrailerSize_;
      }
      // the remaining payload fits only partially, fill whole block
      else {
	memcpy(startOfFedBlocks,fedData_[iFed]+fedSize_[iFed]-remainder,payload);
	remainder-=payload;
	leftspace =0;
      }
    }
    
    //!
    //! no remaining fed data
    //!
    if(remainder==0) {
      
      // loop on feds
      while(iFed<fedN_[iSuperFrag]) {
	
	// if the next header does not fit, jump to following block
	if((int)leftspace<fedHeaderSize_) {
	  frlHeader->segsize-=leftspace;
	  break;
	}
	
	// only for random generated data!
	if (configFeds) {
	  fedh_t *fedHeader  =(fedh_t*)fedData_[iFed];
	  fedHeader->eventid =trigNo;
	  fedHeader->eventid|=0x50000000;
	  fedHeader->sourceid=((fedId_[iSuperFrag][iFed]) << 8) & FED_SOID_MASK;
	}
	
	memcpy(startOfFedBlocks,fedData_[iFed],fedHeaderSize_);
	  
	leftspace       -=fedHeaderSize_;
	startOfFedBlocks+=fedHeaderSize_;
	
	// fed fits with its trailer
	if(fedSize_[iFed]-fedHeaderSize_<=leftspace) {

	  if (configFeds) {
	    fedt_t* fedTrailer=(fedt_t*)(fedData_[iFed]+
					 fedSize_[iFed]-fedTrailerSize_);
	    fedTrailer->eventsize  =fedSize_[iFed];
	    fedTrailer->eventsize /=8; //wc in fed trailer in 64bit words
	    fedTrailer->eventsize |=0xa0000000;
	    fedTrailer->conscheck  =0x0;
	    unsigned short crc=evf::compute_crc(fedData_[iFed],fedSize_[iFed]);
	    fedTrailer->conscheck=(crc<<FED_CRCS_SHIFT);
	  }
	  
	  memcpy(startOfFedBlocks,
		 fedData_[iFed]+fedHeaderSize_,
		 fedSize_[iFed]-fedHeaderSize_);
	  
	  leftspace       -=(fedSize_[iFed]-fedHeaderSize_);
	  startOfFedBlocks+=(fedSize_[iFed]-fedHeaderSize_);
	  
	}
	// fed payload fits only without fed trailer
	else if(fedSize_[iFed]-fedHeaderSize_-fedTrailerSize_<=leftspace) {

	  memcpy(startOfFedBlocks,
		 fedData_[iFed]+fedHeaderSize_,
		 fedSize_[iFed]-fedHeaderSize_-fedTrailerSize_);
	  
	  leftspace         -=(fedSize_[iFed]-fedHeaderSize_-fedTrailerSize_);
	  frlHeader->segsize-=leftspace;
	  fedTrailerLeft     =true;
	  remainder          =fedTrailerSize_;

	  break;
	}
	// fed payload fits only partially
	else {
	  memcpy(startOfFedBlocks,fedData_[iFed]+fedHeaderSize_,leftspace);
	  remainder=fedSize_[iFed]-fedHeaderSize_-leftspace;
	  leftspace=0;
	  
	  break;
	}
	
	// !! increase iFed !!
	iFed++;
	
      } // while (iFed<fedN_)
      
      // earmark the last block
      if (iFed==fedN_[iSuperFrag] && remainder==0 && !last) {
	frlHeader->segsize-=leftspace;
	int msgSize=stdMsg->MessageSize << 2;
	msgSize   -=leftspace;
	bufRef->setDataSize(msgSize);
	stdMsg->MessageSize=msgSize >> 2;
	frlHeader->segsize =frlHeader->segsize | FRL_LAST_SEGM;
	last=true;
      }
      
    } // if (remainder==0)
    
    if(iBlock==0) { // This is the first fragment block / message
      head=bufRef;
      tail=bufRef;
    }
    else {
      tail->setNextReference(bufRef);
      tail=bufRef;
    }
    
    if((iBlock==nBlock-1) && remainder!=0) {
      nBlock++;
      warning=true;
    }
    
  } // for (iBlock)
  
  // fix case where block estimate was wrong
  if(warning) {
    toolbox::mem::Reference* next=head;
    do {
      block =(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)next->getDataLocation();
      block->nbBlocksInSuperFragment=nBlock;		
    } while((next=next->getNextReference()));
  }
  
  return head; // return the top of the chain
}


//______________________________________________________________________________
int BU::estimateNBlocks(unsigned int iSuperFrag,unsigned int fullBlockPayload)
{
  int result(0);
  
  U32 curbSize=frlHeaderSize_;
  U32 totSize =curbSize;
  
  for(unsigned int i=0;i<fedN_[iSuperFrag];i++) {
    curbSize+=fedSize_[i];
    totSize +=fedSize_[i];
    
    // the calculation of the number of blocks needed must handle the
    // fact that RUs can accommodate more than one frl block and
    // remove intermediate headers
    
    if(curbSize > fullBlockPayload) {
      curbSize+=frlHeaderSize_*(curbSize/fullBlockPayload);
      result  +=curbSize/fullBlockPayload;
      
      if(curbSize%fullBlockPayload>0)
	totSize+=frlHeaderSize_*(curbSize/fullBlockPayload);
      else 
	totSize+=frlHeaderSize_*((curbSize/fullBlockPayload)-1);
      
      curbSize=curbSize%fullBlockPayload;
    }
  }	
  
  if(curbSize!=0) result++;
  result=totSize/fullBlockPayload+(totSize%fullBlockPayload>0 ? 1 : 0);
  
  return result;
}


//______________________________________________________________________________
void BU::lock()
{
  lock_.take();
}


//______________________________________________________________________________
void BU::unlock()
{
  lock_.give();
}


//______________________________________________________________________________
void BU::dumpFrame(unsigned char* data,unsigned int len)
{
  char left1[20];
  char left2[20];
  char right1[20];
  char right2[20];

  printf("Byte  0  1  2  3  4  5  6  7\n");
  
  int c(0);
  int pos(0);
  
  for (unsigned int i=0;i<(len/8);i++) {
    int rpos(0);
    int off(3);
    for (pos=0;pos<12;pos+=3) {
      sprintf(&left1[pos],"%2.2x ",
	      ((unsigned char*)data)[c+off]);
      sprintf(&right1[rpos],"%1c",
	      ((data[c+off] > 32)&&(data[c+off] < 127)) ? data[c+off] : '.');
      sprintf (&left2[pos],"%2.2x ",
	       ((unsigned char*)data)[c+off+4]);
      sprintf (&right2[rpos],"%1c",
	       ((data[c+off+4] > 32)&&(data[c+off+4]<127)) ? data[c+off+4] : '.');
      rpos++;
      off--;
    }
    c+=8;
    
    printf ("%4d: %s%s ||  %s%s  %x\n",
	    c-8, left1, left2, right1, right2, (int)&data[c-8]);
  }
  
  fflush(stdout);	
}



////////////////////////////////////////////////////////////////////////////////
// xdaq instantiator implementation macro
////////////////////////////////////////////////////////////////////////////////

XDAQ_INSTANTIATOR_IMPL(BU)
