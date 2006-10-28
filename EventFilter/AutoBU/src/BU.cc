////////////////////////////////////////////////////////////////////////////////
//
// BU
// --
//
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/AutoBU/interface/BU.h"

#include "EventFilter/Utilities/interface/Crc.h"

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
  , fedSizeMax_(16384)
  , mode_("RANDOM")
  , debug_(true)
  , dataBufSize_(32768)
  , nSuperFrag_(64)
  , fedSizeMean_(1024)    // mean  of fed size for rnd generation
  , fedSizeWidth_(1024)   // width of fed size for rnd generation
  , useFixedFedSize_(false)
  , nbMBPerSec_(0.0)
  , memUsedInMB_(0.0)
  , nbEvents_(0)
  , nbEventsPerSec_(0)
  , nbDiscardedEvents_(0)
 , nbEventsLast_(0)
  , nbBytes_(0)
  , i2oPool_(0)
  , bSem_(BSem::FULL)
{
  // initialize application info
  xmlClass_=getApplicationDescriptor()->getClassName();
  instance_=getApplicationDescriptor()->getInstance();
  
  stringstream oss;
  oss<<xmlClass_<<instance_;
  sourceId_=oss.str();
  
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
  bSem_.take();
  gui_->monInfoSpace()->lock();
  
  // number of events per second measurement
  nbEventsPerSec_=nbEvents_-nbEventsLast_;
  nbEventsLast_  =nbEvents_;
  
  // number of MB per second measurement
  nbMBPerSec_=0.000001*nbBytes_;
  nbBytes_   =0;

  gui_->monInfoSpace()->unlock();
  bSem_.give();
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
  // reset counters
  if (0!=gui_) gui_->resetCounters();

  // initialze timer for nbEventsPerSec / nbMBPerSec measurements
  nbEventsLast_=0;
  nbBytes_=0;
  toolbox::task::getTimerFactory()->createTimer(sourceId_);
  toolbox::task::Timer *timer=toolbox::task::getTimerFactory()->getTimer(sourceId_);
  timer->stop();
  
  LOG4CPLUS_INFO(log_,"BU -> CONFIGURED <-");
}


//______________________________________________________________________________
void BU::enableAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  // start timer for nbEventsPerSec / nbMBPerSec measurements
  toolbox::task::Timer *timer =toolbox::task::getTimerFactory()->getTimer(sourceId_);
  if (0!=timer) {
    toolbox::TimeInterval oneSec(1.);
    toolbox::TimeVal      startTime=toolbox::TimeVal::gettimeofday();
    timer->start();
    timer->scheduleAtFixedRate(startTime,this,oneSec,gui_->monInfoSpace(),sourceId_);
  }
  else {
    LOG4CPLUS_WARN(log_,"could't start timer for nbEventsPerSec measurement");
  }
  
  LOG4CPLUS_INFO(log_,"BU -> ENABLED <-");
}


//______________________________________________________________________________
void BU::suspendAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  bSem_.take();
  LOG4CPLUS_INFO(log_,"BU -> SUSPENDED <-");
}


//______________________________________________________________________________
void BU::resumeAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  bSem_.give();
  LOG4CPLUS_INFO(log_,"BU -> RESUMED <-");
}


//______________________________________________________________________________
void BU::haltAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  toolbox::task::Timer *timer =toolbox::task::getTimerFactory()->getTimer(sourceId_);
  if (0!=timer) timer->stop();
  
  LOG4CPLUS_INFO(log_,"BU -> HALTED <-");
}


//______________________________________________________________________________
void BU::nullAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  cout<<"BU::nullAction()"<<endl;
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
  LOG4CPLUS_DEBUG(log_,"received I2O_BU_ALLOCATE request");

  LOG4CPLUS_WARN(log_,"received I2O_BU_ALLOCATE request");

  // check if the BU is enabled
  toolbox::fsm::State currentState=fsm_->getCurrentState();
  if (currentState!='E') {
    LOG4CPLUS_WARN(log_,"Ignore I2O_BU_ALLOCATE while *not* enabled!");
    bufRef->release();
    return;
  }

  bSem_.take();
  
  // process message
  I2O_MESSAGE_FRAME             *stdMsg;
  I2O_BU_ALLOCATE_MESSAGE_FRAME *msg;

  stdMsg=(I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
  msg   =(I2O_BU_ALLOCATE_MESSAGE_FRAME*)stdMsg;

  unsigned int nbEvents=msg->n;
  
  // loop over all requested events
  for(unsigned int i=0;i<nbEvents;i++) {
    
    U32     fuTransactionId=msg   ->allocate[i].fuTransactionId; // assigned by FU
    I2O_TID fuTid          =stdMsg->InitiatorAddress;            // declared to i2o
    
    // if a raw data provider is present, request an event from it
    unsigned int runNumber=0; // not needed
    unsigned int evtNumber=(nbEvents_+1)%0x1000000;
    FEDRawDataCollection* event(0);
    if (0!=PlaybackRawDataProvider::instance()) {
      event=PlaybackRawDataProvider::instance()->getFEDRawData(runNumber,evtNumber);
    }
    
    
    //
    // loop over all superfragments in each event    
    //
    unsigned int nSuperFrag(nSuperFrag_);
    std::vector<unsigned int> validFedIds;
    if (0!=event) {
      for (unsigned int j=0;j<(unsigned int)FEDNumbering::lastFEDId()+1;j++)
	if (event->FEDData(j).size()>0) validFedIds.push_back(j);
      nSuperFrag=validFedIds.size();
    }
    
    if (0==nSuperFrag) {
      LOG4CPLUS_INFO(log_,"No data in FEDRawDataCollection, skip!");
      continue;
    }
    
    for (unsigned int iSuperFrag=0;iSuperFrag<nSuperFrag;iSuperFrag++) {
      
      // "playback", read events from a file
      if (0!=event) {
	if (0==fedN_) initFedBuffers(1);
	fedData_[0]=event->FEDData(validFedIds[iSuperFrag]).data();
	fedSize_[0]=event->FEDData(validFedIds[iSuperFrag]).size();
	LOG4CPLUS_DEBUG(log_,
			"transId="<<fuTransactionId<<": "
			<<"fed "<<validFedIds[iSuperFrag]
			<<" in superfragment "<<iSuperFrag+1<<"/"<<nSuperFrag);
      }
      
      // randomly generate fed data (*including* headers and trailers)
      else {
	generateRndmFEDs();
      }
      
      
      // create super fragment
      toolbox::mem::Reference *superFrag=
	createSuperFrag(fuTid,           // fuTid
			fuTransactionId, // fuTransaction
			evtNumber,       // current trigger (event) number
			iSuperFrag,      // current super fragment
			nSuperFrag       // number of super fragments
			);
      
      if (debug_) debug(superFrag);
      
      I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *frame =
	(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)(superFrag->getDataLocation());
      
      unsigned int msgSizeInBytes=frame->PvtMessageFrame.StdMessageFrame.MessageSize<<2;
      superFrag->setDataSize(msgSizeInBytes);
      nbBytes_.value_+=msgSizeInBytes;
      
      xdaq::ApplicationDescriptor *buAppDesc=
	getApplicationDescriptor();
      
      xdaq::ApplicationDescriptor *fuAppDesc= 
	i2o::utils::getAddressMap()->getApplicationDescriptor(fuTid);
      
      getApplicationContext()->postFrame(superFrag,buAppDesc,fuAppDesc);
    }
    
    if (0!=event) delete event;
    
    nbEvents_.value_++;
  }

  // Free the request message from the FU
  bufRef->release();
  
  bSem_.give();
}


//______________________________________________________________________________
void BU::I2O_BU_COLLECT_Callback(toolbox::mem::Reference *bufRef)
{
  bufRef->release();
  LOG4CPLUS_FATAL(log_,"I2O_BU_COLLECT_Callback() not implemented!");
  exit(-1);
}


//______________________________________________________________________________
void BU::I2O_BU_DISCARD_Callback(toolbox::mem::Reference *bufRef)
{
  // Does nothing but free the incoming I2O message
  nbDiscardedEvents_.value_++;
  bufRef->release();
}


//______________________________________________________________________________
void BU::initFedBuffers(unsigned int fedN)
{
  clearFedBuffers();
  fedN_=fedN;

  fedData_=new unsigned char*[fedN_];
  fedSize_=new unsigned int[fedN_];

  for (unsigned int i=0;i<fedN_;i++) {    
    fedData_[i]=new unsigned char[fedSizeMax_];
    fedSize_[i]=0;
  }
}


//______________________________________________________________________________
void BU::clearFedBuffers()
{
  if (0!=fedData_) {
    for (unsigned int i=0;i<fedN_;i++) delete [] fedData_[i];
    delete [] fedData_;
  }

  if (0!=fedSize_) delete [] fedSize_;

  fedN_=0;
}


//______________________________________________________________________________
void BU::generateRndmFEDs()
{
  // 16 FEDs per superfragment
  if (fedN_<16) initFedBuffers(16);
  
  // size of FEDs
  if(useFixedFedSize_) {
    for (unsigned int i=0;i<16;i++) fedSize_[i]=fedSizeMean_;
  }
  else {
    for(unsigned int i=0;i<fedN_;i++) {
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


//______________________________________________________________________________
void BU::exportParameters()
{
  if (0==gui_) {
    LOG4CPLUS_ERROR(log_,"No GUI, can't export parameters");
    return;
  }
  
  gui_->addMonitorParam("stateName",          &fsm_->stateName_);
  gui_->addMonitorParam("mode",               &mode_);
  gui_->addMonitorParam("debug",              &debug_);
  gui_->addMonitorParam("nbMBPerSec",         &nbMBPerSec_);
  gui_->addMonitorParam("memUsedInMB",        &memUsedInMB_);

  gui_->addStandardParam("dataBufSize",       &dataBufSize_);
  gui_->addStandardParam("nSuperFrag",        &nSuperFrag_);
  gui_->addStandardParam("fedSizeMean",       &fedSizeMean_);
  gui_->addStandardParam("fedSizeWidth",      &fedSizeWidth_);
  gui_->addStandardParam("useFixedFedSize",   &useFixedFedSize_);
  
  gui_->addMonitorCounter("nbEvents",         &nbEvents_);
  gui_->addMonitorCounter("nbEventsPerSec",   &nbEventsPerSec_);
  gui_->addMonitorCounter("nbDiscardedEvents",&nbDiscardedEvents_);

  gui_->exportParameters();

  gui_->addItemRetrieveListener("mode",       this);
  gui_->addItemRetrieveListener("memUsedInMB",this);

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
  bool   configFeds      =(0==PlaybackRawDataProvider::instance());
  size_t msgHeaderSize   =sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
  size_t fullBlockPayload=dataBufSize_-msgHeaderSize;
  
  if((fullBlockPayload%4)!=0)
    LOG4CPLUS_ERROR(log_,"The full block payload of "
		    <<fullBlockPayload<<" bytes is not a multiple of 4");
  
  unsigned int nBlock=estimateNBlocks(fullBlockPayload);
  
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

    block->buResourceId           =fuTransaction; // whatever!?
    block->fuTransactionId        =fuTransaction;
    block->blockNb                =iBlock;
    block->nbBlocksInSuperFragment=nBlock;
    block->superFragmentNb        =iSuperFrag;
    block->nbSuperFragmentsInEvent=nSuperFrag;

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
      if((iFed==(fedN_-1)) && !last) {
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
	if(iFed==fedN_-1) {
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
      while(iFed<fedN_) {
	
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
	  fedHeader->sourceid=((iFed+iSuperFrag*16) << 8) & FED_SOID_MASK;
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
      if (iFed==fedN_ && remainder==0 && !last) {
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
      tail->setNextReference(bufRef); //set link in list
      tail=bufRef;
    }
    
    if((iBlock==nBlock-1) && remainder!=0) {
      nBlock++;
      warning=true;
    }
    
  } // for (iBlock)
  
  // fix case where block estimate was wrong
  if(warning) {
    bufRef=head;
    do {
      stdMsg=(I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
      pvtMsg=(I2O_PRIVATE_MESSAGE_FRAME*)stdMsg;
      block =(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)stdMsg;
      block->nbBlocksInSuperFragment=nBlock;		
    } while((bufRef=bufRef->getNextReference()));
  }
  
  return head; // return the top of the chain
}


//______________________________________________________________________________
int BU::estimateNBlocks(size_t fullBlockPayload)
{
  int result(0);
  
  U32 curbSize=frlHeaderSize_;
  U32 totSize =curbSize;
  
  for(unsigned int i=0;i<fedN_;i++) {
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
int BU::check_event_data(unsigned long* blocks_adrs, int nmb_blocks)
{
  int   retval= 0;
  int   fedid =-1;

  int   feds  =-1; // fed size  
  char* fedd  = 0; //temporary buffer for fed data

  unsigned char* blk_cursor    = 0;
  int            current_trigno=-1;

  int            seglen_left=0;
  int            fed_left   =0;
  
  //loop on blocks starting from last
  for(int iblk=nmb_blocks-1;iblk>=0;iblk--) {
    
    blk_cursor=(unsigned char*)blocks_adrs[iblk];

    frlh_t *ph        =(frlh_t *)blk_cursor;
    int hd_trigno     =ph->trigno;
    int hd_segsize    =ph->segsize;
    int segsize_proper=hd_segsize & ~FRL_LAST_SEGM ;
    
    // check trigno
    if (current_trigno == -1) {
      current_trigno=hd_trigno;
    }
    else {
      if (current_trigno!=hd_trigno) {
	printf("data error nmb_blocks %d iblock %d trigno expect %d got %d \n"
	       ,nmb_blocks,iblk,current_trigno,hd_trigno) ;
	return -1;
      } 
    }
    
    // check that last block flagged as last segment and none of the others
    if (iblk == nmb_blocks-1) {
      if  (!(hd_segsize & FRL_LAST_SEGM)) {
	printf("data error nmb_blocks %d iblock %d last_segm not set \n",
	       nmb_blocks,iblk) ;
	return -1;
      }
    }
    else {
      if ((hd_segsize & FRL_LAST_SEGM)) {
	printf("data error nmb_blocks %d iblock %d last_segm  set \n",
	       nmb_blocks,iblk) ;
	return -1;
      }
    }
    
    blk_cursor += frlHeaderSize_;
    seglen_left = segsize_proper;
    blk_cursor += segsize_proper;
    while(seglen_left>=0) {

      if(fed_left == 0) {
	
	if(feds>=0) {
	  retval += 0;
	  delete[] fedd;
	  feds = -1;
	}
	
	if(seglen_left==0)break;
	
	seglen_left-=fedTrailerSize_;
	blk_cursor -=fedTrailerSize_;
	fedt_t *pft =(fedt_t*)blk_cursor;
	int fedlen  =pft->eventsize & FED_EVSZ_MASK;
	fedlen     *=8; // in the fed trailer, wc is in 64 bit words
	
	feds=fedlen-fedHeaderSize_-fedTrailerSize_;
	fedd=new char[feds];

	if((seglen_left-(fedlen-fedTrailerSize_)) >= 0) {
	  blk_cursor-=feds;
	  memcpy(fedd,blk_cursor,feds);
	  seglen_left-=(fedlen-fedTrailerSize_);
	  fed_left=0;
	  blk_cursor-=fedHeaderSize_;
	  fedh_t *pfh=(fedh_t *)blk_cursor;
	  fedid=pfh->sourceid & FED_SOID_MASK;
	  fedid=fedid >> 8;
	  
	  // DEBUG
	  if((pfh->eventid & FED_HCTRLID_MASK)!=0x50000000)
	    cout<<"check_event_data (1): fedh error! trigno="<<hd_trigno
		<<" fedid="<<fedid<<endl;
	  // END DEBUG
	}
	else {
	  blk_cursor=(unsigned char*)blocks_adrs[iblk]+frlHeaderSize_;
	  fed_left  =fedlen-fedTrailerSize_-seglen_left;
	  memcpy(fedd+feds-seglen_left,blk_cursor,seglen_left);
	  seglen_left=0;
	}
      }
      else if(fed_left > fedHeaderSize_) {
	if(seglen_left==0)break;
	if(seglen_left-fed_left >= 0) {
	  blk_cursor-=(fed_left-fedHeaderSize_);
	  memcpy(fedd,blk_cursor,fed_left-fedHeaderSize_);
	  seglen_left-=fed_left;
	  blk_cursor -=fedHeaderSize_;
	  fed_left    =0;
	  fedh_t *pfh =(fedh_t *)blk_cursor;
	  fedid=pfh->sourceid & FED_SOID_MASK;
	  fedid=fedid >> 8;
	  
	  // DEBUG
	  if((pfh->eventid & FED_HCTRLID_MASK)!=0x50000000)
	    cout<<"check_event_data (2): fedh error! trigno="<<hd_trigno
		<<" fedid="<<fedid<<endl;
	  // END DEBUG
	}
	else if(seglen_left-fed_left+fedHeaderSize_>0) {
	  blk_cursor=(unsigned char*)blocks_adrs[iblk]+frlHeaderSize_;
	  memcpy(fedd,blk_cursor,fed_left-fedHeaderSize_);
	  fed_left=fedHeaderSize_;
	  seglen_left=0;
	}
	else {
	  blk_cursor=(unsigned char*)blocks_adrs[iblk]+frlHeaderSize_;
	  memcpy(fedd+fed_left-fedHeaderSize_-seglen_left,blk_cursor,seglen_left);
	  fed_left-=seglen_left;
	  seglen_left=0;
	}
      }
      else {
	if(seglen_left==0)break;
	blk_cursor-=fedHeaderSize_;
	fed_left=0;
	seglen_left-=fedHeaderSize_;
	fedh_t *pfh=(fedh_t *)blk_cursor;
	fedid=pfh->sourceid & FED_SOID_MASK;
	fedid=fedid >> 8;
      }
    }
    
    //dumpFrame((unsigned char*)blocks_adrs[iblk],segsize_proper+frlHeaderSize_);
  }

  return retval;
}


//______________________________________________________________________________
void BU::debug(toolbox::mem::Reference* ref)
{
  vector<toolbox::mem::Reference*> chain;
  toolbox::mem::Reference *nn = ref;
  chain.push_back(ref);
  int ind = 1;
  while((nn=nn->getNextReference())!=0) {
    chain.push_back(nn);
    ind++;
  }
	  
  //unsigned long blocks_adrs[chain.size()];
  unsigned long* blocks_adrs=new unsigned long[chain.size()];
  for(unsigned int i=0;i<chain.size();i++) {
    blocks_adrs[i]=(unsigned long)chain[i]->getDataLocation()+
      sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
  }
  
  // call method to unwind data structure and check H/T content 
  int ierr=check_event_data(blocks_adrs,chain.size());
  if(ierr!=0) cerr<<"ERROR::check_event_data, code = "<<ierr<<endl;
  delete [] blocks_adrs;
}


//______________________________________________________________________________
void BU::dumpFrame(unsigned char* data,unsigned int len)
{
  //PI2O_MESSAGE_FRAME  ptrFrame = (PI2O_MESSAGE_FRAME)data;
  //printf ("\nMessageSize: %d Function %d\n",
  //ptrFrame->MessageSize,ptrFrame->Function);
  
  char left1[20];
  char left2[20];
  char right1[20];
  char right2[20];
  
  //LOG4CPLUS_ERROR(log_,
  //  toolbox::toString("Byte  0  1  2  3  4  5  6  7\n"));
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
    
    //LOG4CPLUS_ERROR(log_,
    //  toolbox::toString("%4d: %s  ||  %s \n", c-8, left, right));
    printf ("%4d: %s%s ||  %s%s  %x\n",
	    c-8, left1, left2, right1, right2, (int)&data[c-8]);
  }
  
  fflush(stdout);	
}



////////////////////////////////////////////////////////////////////////////////
// xdaq instantiator implementation macro
////////////////////////////////////////////////////////////////////////////////

XDAQ_INSTANTIATOR_IMPL(BU)
