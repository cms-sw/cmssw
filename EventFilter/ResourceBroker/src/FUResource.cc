////////////////////////////////////////////////////////////////////////////////
//
// FUResource
// ----------
//
//            12/10/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ResourceBroker/interface/FUResource.h"
#include "FWCore/Utilities/interface/CRC16.h"
#include "EventFilter/Utilities/interface/GlobalEventNumber.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "interface/shared/frl_header.h"
#include "interface/shared/fed_header.h"
#include "interface/shared/fed_trailer.h"
#include "interface/shared/i2oXFunctionCodes.h"
#include "interface/evb/i2oEVBMsgs.h"

#include "toolbox/mem/Reference.h"

#include "xcept/tools.h"

#include <sstream>
#include <sys/shm.h>


#define FED_HCTRLID    0x50000000
#define FED_TCTRLID    0xa0000000
#define REAL_SOID_MASK 0x0003FF00
#define FED_RBIT_MASK  0x00000004


using namespace std;
using namespace evf;


////////////////////////////////////////////////////////////////////////////////
// initialize static members
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
bool FUResource::doFedIdCheck_ = true;
bool FUResource::useEvmBoard_ = true;
unsigned int FUResource::gtpEvmId_ =  FEDNumbering::getTriggerGTPFEDIds().first;
unsigned int FUResource::gtpDaqId_ =  FEDNumbering::getTriggerGTPFEDIds().second;

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUResource::FUResource(UInt_t fuResourceId,log4cplus::Logger logger)
  : log_(logger)
  , fuResourceId_(fuResourceId)
  , superFragHead_(0)
  , superFragTail_(0)
  , nbBytes_(0)
  , superFragSize_(0)
{
  release();
}


//______________________________________________________________________________
FUResource::~FUResource()
{
  
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void FUResource::allocate(FUShmRawCell* shmCell)
{
  //release();
  shmCell_=shmCell;
  shmCell_->clear();
  shmCell_->setFuResourceId(fuResourceId_);
  eventPayloadSize_=shmCell_->payloadSize();
  nFedMax_         =shmCell_->nFed();
  nSuperFragMax_   =shmCell_->nSuperFrag();
}


//______________________________________________________________________________
void FUResource::release()
{
  doCrcCheck_   =false;
  fatalError_   =false;
  
  buResourceId_ =0xffffffff;
  evtNumber_    =0xffffffff;
  
  if (0!=superFragHead_) {
    try {
      superFragHead_->release();
    }
    catch (xcept::Exception& e) {
      LOG4CPLUS_ERROR(log_,"Failed to release superFragHead: "
		      <<xcept::stdformat_exception_history(e));
    }
  }
  
  superFragHead_=0;
  superFragTail_=0;
  
  iBlock_       =0;
  nBlock_       =0xffffffff;
  iSuperFrag_   =0;
  nSuperFrag_   =0xffffffff;

  nbSent_       =0;
  
  nbErrors_     =0;
  nbCrcErrors_  =0;

  for (UInt_t i=0;i<1024;i++) fedSize_[i]=0;
  eventSize_    =0;
  
  if (0!=shmCell_) {
    shmdt(shmCell_);
    shmCell_=0;
  }
}


//______________________________________________________________________________
void FUResource::process(MemRef_t* bufRef)
{
  if (fatalError()) {
    LOG4CPLUS_WARN(log_,"THIS SHOULD *NEVER* HAPPEN!."); // DEBUG
    bufRef->release();
    return;
  }
  
  MemRef_t* itBufRef = bufRef;
  while(0!=itBufRef&&!fatalError()) {
    MemRef_t* next=itBufRef->getNextReference();
    itBufRef->setNextReference(0);
    
    try {
      processDataBlock(itBufRef);
    }
    catch (xcept::Exception& e) {
      LOG4CPLUS_ERROR(log_,"EVENT LOST:"
		      <<xcept::stdformat_exception_history(e));
      fatalError_=true;
      bufRef->setNextReference(next);
    }
    
    itBufRef=next;
  }
  
  return;
}


//______________________________________________________________________________
void FUResource::processDataBlock(MemRef_t* bufRef)
  throw (evf::Exception)
{
  // reset iBlock_/nBlock_ counters
  if (iBlock_==nBlock_) {
    iBlock_=0;
    nBlock_=0xffffffff;
  }
  
  I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block=
    (I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)bufRef->getDataLocation();
  
  UInt_t iBlock      =block->blockNb;
  UInt_t nBlock      =block->nbBlocksInSuperFragment;
  UInt_t iSuperFrag  =block->superFragmentNb;
  UInt_t nSuperFrag  =block->nbSuperFragmentsInEvent;
  
  UInt_t fuResourceId=block->fuTransactionId;
  UInt_t buResourceId=block->buResourceId;
  UInt_t evtNumber   =block->eventNumber;

  // check fuResourceId consistency
  if (fuResourceId!=fuResourceId_) {
    nbErrors_++;
    stringstream oss;
    oss<<"RU/FU fuResourceId mismatch."
       <<" Received:"<<fuResourceId
       <<" Expected:"<<fuResourceId_;
    XCEPT_RAISE(evf::Exception,oss.str());
  }
  
  // check iBlock consistency
  if (iBlock!=iBlock_) {
    nbErrors_++;
    stringstream oss;
    oss<<"RU/FU block number mismatch."
       <<" Received:"<<iBlock
       <<" Expected:"<<iBlock_;
    XCEPT_RAISE(evf::Exception,oss.str());
  }
  
  // check iSuperFrag consistency
  if (iSuperFrag!=iSuperFrag_) {
    nbErrors_++;
    stringstream oss;
    oss<<"RU/FU superfragment number mismatch."
       <<" Received:"<<iSuperFrag
       <<" Expected:"<<iSuperFrag_;
    XCEPT_RAISE(evf::Exception,oss.str());
  }

  // assign nBlock_
  if (iBlock==0) {
    nBlock_=nBlock;
  }
  else {
    // check nBlock_
    if (nBlock!=nBlock_) {
      nbErrors_++;
      stringstream oss;
      oss<<"RU/FU number of blocks mismatch."
	 <<" Received:"<<nBlock
	 <<" Expected:"<<nBlock_;
      XCEPT_RAISE(evf::Exception,oss.str());
    }
  }
  
  
  // if this is the first block in the event,
  // *assign* evtNumber,buResourceId,nSuperFrag ...
  if (iBlock==0&&iSuperFrag==0) {
    evtNumber_   =evtNumber;
    buResourceId_=buResourceId;
    nSuperFrag_  =nSuperFrag;
    
    shmCell_->setEvtNumber(evtNumber);
    shmCell_->setBuResourceId(buResourceId);

    // check that buffers are allocated for nSuperFrag superfragments
    if(nSuperFrag_>nSuperFragMax_) {
      nbErrors_++;
      stringstream oss;
      oss<<"Invalid maimum number of superfragments."
	 <<" fuResourceId:"<<fuResourceId_
	 <<" evtNumber:"<<evtNumber_
	 <<" nSuperFrag:"<<nSuperFrag_
	 <<" nSuperFragMax:"<<nSuperFragMax_;
      XCEPT_RAISE(evf::Exception,oss.str());
    }
  }
  // ... otherwise,
  // *check* evtNumber,buResourceId,nSuperFrag
  else {
    // check evtNumber
    if (evtNumber!=evtNumber_) {
      nbErrors_++;
      stringstream oss;
      oss<<"RU/FU evtNumber mismatch."
	 <<" Received:"<<evtNumber
	 <<" Expected:"<<evtNumber_;
      XCEPT_RAISE(evf::Exception,oss.str());
    }
    
    // check buResourceId
    if (buResourceId!=buResourceId_) {
      nbErrors_++;
      stringstream oss;
      oss<<"RU/FU buResourceId mismatch."
	 <<" Received:"<<buResourceId
	 <<" Expected:"<<buResourceId_;
      XCEPT_RAISE(evf::Exception,oss.str());
    }
    
    // check nSuperFrag
    if (nSuperFrag!=nSuperFrag_) {
      nbErrors_++;
      stringstream oss;
      oss<<"RU/FU number of superfragments mismatch."
	 <<" Received:"<<nSuperFrag
	 <<" Expected:"<<nSuperFrag_;
      XCEPT_RAISE(evf::Exception,oss.str());
    }
  }
  
  
  // check payload
  try {
    checkDataBlockPayload(bufRef);
  }
  catch (xcept::Exception& e) {
    stringstream oss;
    oss<<"data block payload failed check."
       <<" evtNumber:"<<evtNumber_
       <<" buResourceId:"<<buResourceId_
       <<" iSuperFrag:"<<iSuperFrag_;
    XCEPT_RETHROW(evf::Exception,oss.str(),e);
  }
  
  appendBlockToSuperFrag(bufRef);

  // increment iBlock_, as expected for the next message
  iBlock_++;
  
  // superfragment complete ...
  bool lastBlockInSuperFrag=(iBlock==nBlock-1);
  if (lastBlockInSuperFrag) {
    
    // ... fill the FED buffers contained in the superfragment
    try {
      superFragSize();
      fillSuperFragPayload();
      findFEDs();
    }
    catch (xcept::Exception& e) {
      stringstream oss;
      oss<<"Invalid super fragment."
	 <<" evtNumber:"<<evtNumber_
	 <<" buResourceId:"<<buResourceId_
	 <<" iSuperFrag:"<<iSuperFrag_;
      XCEPT_RETHROW(evf::Exception,oss.str(),e);
    }
    
    // ... release the buffers associated with the superfragment
    try {
      releaseSuperFrag();
    }
    catch (xcept::Exception& e) {
      nbErrors_++;
      stringstream oss;
      oss<<"Failed to release super fragment."
	 <<" evtNumber:"<<evtNumber_
	 <<" buResourceId:"<<buResourceId_
	 <<" iSuperFrag:"<<iSuperFrag_;
      XCEPT_RETHROW(evf::Exception,oss.str(),e);
    }

    // increment iSuperFrag_, as expected for the next message(s)
    iSuperFrag_++;
    
  } // lastBlockInSuperFragment
  
  return;
}


//______________________________________________________________________________
void FUResource::checkDataBlockPayload(MemRef_t* bufRef)
  throw (evf::Exception)
{
  UInt_t   frameSize      =0;
  UInt_t   bufSize        =0;
  UInt_t   segSize        =0;
  UInt_t   segSizeExpected=0;

  frlh_t  *frlHeader      =0;
  
  UChar_t *blockAddr      =0;
  UChar_t *frlHeaderAddr  =0;
  
  frameSize    =sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);

  blockAddr    =(UChar_t*)bufRef->getDataLocation();
  frlHeaderAddr=blockAddr+frameSize;
  frlHeader    =(frlh_t*)frlHeaderAddr;
  
  I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block
    =(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)blockAddr;
    

  // check that FRL trigno is consistent with FU evtNumber
  if(evtNumber_!=frlHeader->trigno) {
    nbErrors_++;
    std::stringstream oss;
    oss<<"FRL header \"trigno\" does not match "
       <<"FU  \"evtNumber\"."
       <<" trigno:"<<frlHeader->trigno
       <<" evtNumber:"<<evtNumber_;
    XCEPT_RAISE(evf::Exception,oss.str());
  }
  

  // check that FRL trigno is consistent with RU eventNumber
  if(block->eventNumber!=frlHeader->trigno) {
    nbErrors_++;
    std::stringstream oss;
    oss<<"FRL header \"trigno\" does not match "
       <<"RU builder header \"eventNumber\"."
       <<" trigno:"<<frlHeader->trigno
       <<" eventNumber:"<<block->eventNumber;
    XCEPT_RAISE(evf::Exception,oss.str());
  }
  

  // check that block numbers reported by FRL / RU are consistent
  if(block->blockNb!=frlHeader->segno) {
    nbErrors_++;
    std::stringstream oss;
    oss<<"FRL header \"segno\" does not match"
       <<"RU builder header \"blockNb\"."
       <<" segno:"<<frlHeader->segno
       <<" blockNb:"<<block->blockNb;
    XCEPT_RAISE(evf::Exception,oss.str());
  }
  
  
  // reported block number consistent with expectation
  if(block->blockNb!=iBlock_) {
    nbErrors_++;
    std::stringstream oss;
    oss<<"Incorrect block number."
       <<" Expected:"<<iBlock_
       <<" Received:"<<block->blockNb;
    XCEPT_RAISE(evf::Exception, oss.str());
  }
  
  
  // reported payload size consistent with expectation
  bufSize        =bufRef->getDataSize();
  segSizeExpected=bufSize-frameSize-sizeof(frlh_t);
  segSize        =frlHeader->segsize & FRL_SEGSIZE_MASK;
  if(segSize!=segSizeExpected) {
    nbErrors_++;
    std::stringstream oss;
    oss<<"FRL header segment size is not as expected."
       <<" Expected:"<<segSizeExpected
       <<" Received:"<<segSize;
    XCEPT_RAISE(evf::Exception, oss.str());
  }
  
  
  // Check that FU and FRL headers agree on end of super-fragment
  bool fuLastBlockInSuperFrag =(block->blockNb==(block->nbBlocksInSuperFragment-1));
  bool frlLastBlockInSuperFrag=((frlHeader->segsize & FRL_LAST_SEGM)!=0);
  if (fuLastBlockInSuperFrag!=frlLastBlockInSuperFrag) {
    nbErrors_++;
    std::stringstream oss;
    oss<<"FU / FRL header end-of-superfragment mismatch."
       <<" FU header:"<<fuLastBlockInSuperFrag
       <<" FRL header:"<<frlLastBlockInSuperFrag;
    XCEPT_RAISE(evf::Exception,oss.str());
  }
  
  return;
}


//______________________________________________________________________________
void FUResource::appendBlockToSuperFrag(MemRef_t* bufRef)
{
  if (0==superFragHead_) {
    superFragHead_=bufRef;
    superFragTail_=bufRef;
  }
  else {
    superFragTail_->setNextReference(bufRef);
    superFragTail_=bufRef;
  }
  return;
}


//______________________________________________________________________________
void FUResource::superFragSize() throw (evf::Exception)
{
  UChar_t *blockAddr    =0;
  UChar_t *frlHeaderAddr=0;
  frlh_t  *frlHeader    =0;

  superFragSize_=0;

  UInt_t frameSize=sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
  MemRef_t* bufRef=superFragHead_;
  
  while (0!=bufRef) {
    blockAddr      =(UChar_t*)bufRef->getDataLocation();
    frlHeaderAddr  =blockAddr+frameSize;
    frlHeader      =(frlh_t*)frlHeaderAddr;
    superFragSize_+=frlHeader->segsize & FRL_SEGSIZE_MASK; 
    bufRef         =bufRef->getNextReference();
  }
  
  eventSize_+=superFragSize_;

  if (eventSize_>eventPayloadSize_) {  
    nbErrors_++;
    stringstream oss;
    oss<<"Event size exceeds maximum size."
       <<" fuResourceId:"<<fuResourceId_
       <<" evtNumber:"<<evtNumber_
       <<" iSuperFrag:"<<iSuperFrag_
       <<" eventSize:"<<eventSize_
       <<" eventPayloadSize:"<<eventPayloadSize_;
    XCEPT_RAISE(evf::Exception,oss.str());
  }
  
}


//______________________________________________________________________________
void FUResource::fillSuperFragPayload() throw (evf::Exception)
{
  UChar_t *blockAddr    =0;
  UChar_t *frlHeaderAddr=0;
  UChar_t *fedAddr      =0;
  UInt_t   nbBytes      =0;
  UInt_t   nbBytesTot   =0;
  frlh_t  *frlHeader    =0;
  UChar_t *bufferPos    =0;
  UChar_t *startPos     =0;
  
  MemRef_t* bufRef=superFragHead_;
  while(bufRef != 0) {
    blockAddr    =(UChar_t*)bufRef->getDataLocation();
    frlHeaderAddr=blockAddr+sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
    fedAddr      =frlHeaderAddr+sizeof(frlh_t);
    frlHeader    =(frlh_t*)frlHeaderAddr;
    nbBytes      =frlHeader->segsize & FRL_SEGSIZE_MASK;
    nbBytesTot  +=nbBytes;
    
    // check if still within limits
    if(nbBytesTot>superFragSize_) {
      nbErrors_++;
      stringstream oss;
      oss<<"Reached end of buffer."
	 <<" fuResourceId:"<<fuResourceId_
	 <<" evtNumber:"<<evtNumber_
	 <<" iSuperFrag:"<<iSuperFrag_;
      XCEPT_RAISE(evf::Exception,oss.str());
    }
    
    bufferPos=shmCell_->writeData(fedAddr,nbBytes);
    if (0==startPos) startPos=bufferPos;
    
    nbBytes_+=nbBytes;
    bufRef=bufRef->getNextReference();
  }
  
  if (!shmCell_->markSuperFrag(iSuperFrag_,superFragSize_,startPos)) {
    nbErrors_++;
    stringstream oss;
    oss<<"Failed to mark super fragment in shared mem buffer."
       <<" fuResourceId:"<<fuResourceId_
       <<" evtNumber:"<<evtNumber_
       <<" iSuperFrag:"<<iSuperFrag_;
    XCEPT_RAISE(evf::Exception,oss.str());
  }
  
  return;
}


//______________________________________________________________________________
void FUResource::findFEDs() throw (evf::Exception)
{
  UChar_t* superFragAddr =0;
  UInt_t   superFragSize =0;
  
  UChar_t *fedTrailerAddr=0;
  UChar_t *fedHeaderAddr =0;
  
  UInt_t   fedSize       =0;
  UInt_t   sumOfFedSizes =0;
  UInt_t   evtNumber     =0;
  
  UShort_t crc           =0;
  UShort_t crcChk        =0;
  
  fedt_t  *fedTrailer    =0;
  fedh_t  *fedHeader     =0;
  
  
  superFragAddr =shmCell_->superFragAddr(iSuperFrag_);
  superFragSize =shmCell_->superFragSize(iSuperFrag_);
  fedTrailerAddr=superFragAddr+superFragSize-sizeof(fedt_t);
  
  while (fedTrailerAddr>superFragAddr) {
    
    fedTrailer    =(fedt_t*)fedTrailerAddr;
    fedSize       =(fedTrailer->eventsize & FED_EVSZ_MASK) << 3;
    sumOfFedSizes+=fedSize;
    
    // check for fed trailer id
    if ((fedTrailer->eventsize & FED_TCTRLID_MASK)!=FED_TCTRLID) {
      nbErrors_++;
      stringstream oss;
      oss<<"Missing FED trailer id."
	 <<" evtNumber:"<<evtNumber_
	 <<" iSuperFrag:"<<iSuperFrag_;
      XCEPT_RAISE(evf::Exception,oss.str());
    }
    
    fedHeaderAddr=fedTrailerAddr-fedSize+sizeof(fedt_t);
    
    // check that fed header is within buffer
    if(fedHeaderAddr<superFragAddr) {
      nbErrors_++;
      stringstream oss;
      oss<<"FED header address out-of-bounds."
	 <<" evtNumber:"<<evtNumber_
	 <<" iSuperFrag:"<<iSuperFrag_;
      XCEPT_RAISE(evf::Exception,oss.str());
    }
    
    // check that payload starts within buffer
    if((fedHeaderAddr+sizeof(fedh_t))>(superFragAddr+superFragSize)) {
      nbErrors_++;
      stringstream oss;
      oss<<"FED payload out-of-bounds."
	 <<" evtNumber:"<<evtNumber_
	 <<" iSuperFrag:"<<iSuperFrag_;
      XCEPT_RAISE(evf::Exception,oss.str());
    }
    
    fedHeader  =(fedh_t*)fedHeaderAddr;
    
    // check for fed header id
    if ((fedHeader->eventid & FED_HCTRLID_MASK)!=FED_HCTRLID) {
      nbErrors_++;
      stringstream oss;
      oss<<"Missing FED header id."
	 <<" evtNumber:"<<evtNumber_
	 <<" iSuperFrag:"<<iSuperFrag_;
      XCEPT_RAISE(evf::Exception,oss.str());
    }
    
    UInt_t fedId=(fedHeader->sourceid & REAL_SOID_MASK) >> 8;
    
    // check evtNumber consisency
    evtNumber=fedHeader->eventid & FED_LVL1_MASK;
    if (evtNumber!=evtNumber_) {
      nbErrors_++;
      stringstream oss;
      oss<<"FU / FED evtNumber mismatch."
	 <<" FU:"<<evtNumber_
	 <<" FED:"<<evtNumber
	 <<" fedid:"<<fedId;
      XCEPT_RAISE(evf::Exception,oss.str());
    }
  
    // check that fedid is within valid ranges
    if (fedId>=1024||
	(doFedIdCheck_&&(!FEDNumbering::inRange(fedId)))) {
      LOG4CPLUS_WARN(log_,"Invalid fedid. Data will still be logged"
		     <<" evtNumber:"<<evtNumber_
		     <<" fedid:"<<fedId);
      nbErrors_++;
    }

    // check if a previous fed has already claimed same fed id

    if(fedSize_[fedId]!=0) {
      LOG4CPLUS_ERROR(log_,"Duplicated fedid. Data will be lost for"
		      <<" evtNumber:"<<evtNumber_
		      <<" fedid:"<<fedId);
      nbErrors_++;
    }
    
    if (fedId<1024) fedSize_[fedId]=fedSize;

    //if gtp EVM block is available set cell event number to global partition-independent trigger number
    //daq block partition-independent event number is left as an option in case of problems

    if(useEvmBoard_ && (fedId == gtpEvmId_))
      if(evf::evtn::evm_board_sense(fedHeaderAddr)) shmCell_->setEvtNumber(evf::evtn::get(fedHeaderAddr, true));
    if(!useEvmBoard_ && (fedId == gtpDaqId_))
      if(evf::evtn::daq_board_sense(fedHeaderAddr)) shmCell_->setEvtNumber(evf::evtn::get(fedHeaderAddr, false));
    // crc check
    if (doCrcCheck_) {
      UInt_t conscheck=fedTrailer->conscheck;
      crc=((fedTrailer->conscheck & FED_CRCS_MASK) >> FED_CRCS_SHIFT);
      fedTrailer->conscheck &= (~FED_CRCS_MASK);
      fedTrailer->conscheck &= (~FED_RBIT_MASK);
      crcChk=compute_crc(fedHeaderAddr,fedSize);
      
      if (crc!=crcChk) {
	LOG4CPLUS_INFO(log_,"crc check failed."
		       <<" evtNumber:"<<evtNumber_
		       <<" fedid:"<<fedId
		       <<" crc:"<<crc
		       <<" chk:"<<crcChk);
	nbErrors_++;
	nbCrcErrors_++;
      }
      fedTrailer->conscheck=conscheck;
    }
    
    
    // mark fed
    if (!shmCell_->markFed(fedId,fedSize,fedHeaderAddr)) {
      nbErrors_++;
      stringstream oss;
      oss<<"Failed to mark fed in buffer."
	 <<" evtNumber:"<<evtNumber_
	 <<" fedId:"<<fedId
	 <<" fedSize:"<<fedSize
	 <<" fedAddr:0x"<<hex<<(int)fedHeaderAddr<<dec;
      XCEPT_RAISE(evf::Exception,oss.str());
    }
    
    // Move to the next fed trailer
    fedTrailerAddr=fedTrailerAddr-fedSize;
  }
  
  // check that we indeed end up on the starting address of the buffer
  if ((fedTrailerAddr+sizeof(fedh_t))!=superFragAddr) {
    std::stringstream oss;
    oss<<"First FED in superfragment ouf-of-bound."
       <<" evtNumber:"<<evtNumber_
       <<" iSuperFrag:"<<iSuperFrag_;
    XCEPT_RAISE(evf::Exception,oss.str());
  }
  
  return;
}


//______________________________________________________________________________
void FUResource::releaseSuperFrag()
{
  if (0==superFragHead_) return;
  superFragHead_->release(); // throws xcept::Exception
  superFragHead_=0;
  superFragTail_=0;
  return;
}


//______________________________________________________________________________
UInt_t FUResource::nbErrors(bool reset)
{
  UInt_t result=nbErrors_;
  if (reset) nbErrors_=0;
  return result;
}


//______________________________________________________________________________
UInt_t FUResource::nbCrcErrors(bool reset)
{
  UInt_t result=nbCrcErrors_;
  if (reset) nbCrcErrors_=0;
  return result;
}


//______________________________________________________________________________
UInt_t FUResource::nbBytes(bool reset)
{
  UInt_t result=nbBytes_;
  if (reset) nbBytes_=0;
  return result;
}
