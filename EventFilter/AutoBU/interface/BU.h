#ifndef __BU_h__
#define __BU_h__


#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/Playback/interface/PlaybackRawDataProvider.h"

#include "xdaq/include/xdaq/WebApplication.h"

#include "toolbox/include/toolbox/mem/HeapAllocator.h"
#include "toolbox/include/toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/include/toolbox/net/URN.h"

#include "xdata/include/xdata/UnsignedInteger32.h"
#include "xdata/include/xdata/Boolean.h"
#include "xdata/include/xdata/String.h"

#include "interface/evb/include/i2oEVBMsgs.h"
#include "interface/shared/include/i2oXFunctionCodes.h"

#include "interface/shared/include/frl_header.h"
#include "interface/shared/include/fed_header.h"
#include "interface/shared/include/fed_trailer.h"

#include "i2o/include/i2o/Method.h"
#include "i2o/utils/include/i2o/utils/AddressMap.h"

#include "CLHEP/Random/RandGauss.h"

#include "extern/i2o/include/i2o/i2oDdmLib.h"


#include <vector>
#include <cmath>


class BU : public xdaq::WebApplication
{
public:
  //
  // typedefs
  //
  typedef std::vector<unsigned char*> ucharvec_t;
  typedef std::vector<unsigned int>   uintvec_t;
  
  
  //
  // xdaq instantiator macro
  //
  XDAQ_INSTANTIATOR();
  
  
  //
  // construction/destruction
  //
  BU(xdaq::ApplicationStub *s);
  virtual ~BU();
  
  
  //
  // member functions
  //
  
  // i2o callbacks
  void buAllocateNMsg(toolbox::mem::Reference *bufRef);
  void buCollectMsg(toolbox::mem::Reference *bufRef);
  void buDiscardNMsg(toolbox::mem::Reference *bufRef);
  
  // generate N dummy FEDs
  void generateNFedFragments(double     fedSizeMean,
			     double     fedSizeWidth,
			     uintvec_t& fedSize);
  
  //estimate number of blocks needed for a superfragment
  int estimateNBlocks(const uintvec_t& fedSize,size_t fullBlockPayload);
  
  // create a supefragment
  toolbox::mem::Reference *createSuperFrag(const I2O_TID& fuTid,
					   const U32&     fuTransaction,
					   const U32&     trigNo,
					   const U32&     iSuperFrag,
					   const U32&     nSuperFrag);
  
  // debug functionality
  void debug(toolbox::mem::Reference* ref);
  int  check_event_data(unsigned long* blocks_adrs,int nmb_blocks);
  void dumpFrame(unsigned char* data,unsigned int len);
  
  
private:
  //
  // member data
  //
  ucharvec_t                fedData_;
  uintvec_t                 fedSize_;

  xdata::UnsignedInteger32  dataBufSize_;
  xdata::UnsignedInteger32  nSuperFrag_;
  xdata::UnsignedInteger32  nbEventsSent_;
  xdata::UnsignedInteger32  nbEventsDiscarded_;
  
  xdata::UnsignedInteger32  fedSizeMean_;
  xdata::UnsignedInteger32  fedSizeWidth_;
  xdata::Boolean            useFixedFedSize_;
  
  toolbox::mem::Pool*       pool_;
  
  
  //
  // static member data
  //
  static const int frlHeaderSize_ =sizeof(frlh_t);
  static const int fedHeaderSize_ =sizeof(fedh_t);
  static const int fedTrailerSize_=sizeof(fedt_t);
  
}; // class BU



/////////////////////////////////////////////////////////////////////////////////
// implementation of inline functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
inline
void BU::buAllocateNMsg(toolbox::mem::Reference *bufRef)
{
  LOG4CPLUS_DEBUG(getApplicationLogger(),"received buAllocate request");

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
    unsigned int evtNumber=(nbEventsSent_+1)%0x1000000;
    FEDRawDataCollection* event(0);
    if (0!=PlaybackRawDataProvider::instance())
      event=PlaybackRawDataProvider::instance()->getFEDRawData(runNumber,evtNumber);
    

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
      LOG4CPLUS_INFO(getApplicationLogger(),"no data in FEDRawDataCollection!Skip");
      continue;
    }
    
    for (unsigned int iSuperFrag=0;iSuperFrag<nSuperFrag;iSuperFrag++) {
      
      // "playback", read events from a file
      if (0!=event) {
	fedData_.clear();
	fedSize_.clear();
	fedData_.push_back(event->FEDData(validFedIds[iSuperFrag]).data());
	fedSize_.push_back(event->FEDData(validFedIds[iSuperFrag]).size());
	LOG4CPLUS_DEBUG(getApplicationLogger(),
			"transId="<<fuTransactionId<<": "
			<<"fed "<<validFedIds[iSuperFrag]
			<<" in superfragment "<<iSuperFrag+1<<"/"<<nSuperFrag);
      }
      
      // randomly generate fed data (*including* headers and trailers)
      else {
	fedData_.resize(16);
	fedSize_.resize(16);
	
	if(useFixedFedSize_) fedSize_.assign(fedSize_.size(),fedSizeMean_);
	else generateNFedFragments((double)fedSizeMean_,
				   (double)fedSizeWidth_,
				   fedSize_);
	for (unsigned int iFed=0;iFed<fedData_.size();iFed++)
	  fedData_[iFed]=new unsigned char[fedSize_[iFed]];
      }
      
      
      // create super fragment
      toolbox::mem::Reference *superFrag=
	createSuperFrag(fuTid,           // fuTid
			fuTransactionId, // fuTransaction
			evtNumber,       // current trigger (event) number
			iSuperFrag,      // current super fragment
			nSuperFrag       // number of super fragments
			);
      
      debug(superFrag);
      
      // clean up randomly generated data
      if (0==event) {
	for (unsigned int iFed=0;iFed<fedData_.size();iFed++)
	  delete [] fedData_[iFed];
      }
      
      I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *frame =
	(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)(superFrag->getDataLocation());
      
      superFrag->setDataSize(frame->PvtMessageFrame.StdMessageFrame.MessageSize<<2);
      
      xdaq::ApplicationDescriptor *buAppDesc=
	getApplicationDescriptor();
      
      xdaq::ApplicationDescriptor *fuAppDesc= 
	i2o::utils::getAddressMap()->getApplicationDescriptor(fuTid);
      
      getApplicationContext()->postFrame(superFrag,buAppDesc,fuAppDesc);
    }
    
    nbEventsSent_.value_++;
  }

  // Free the request message from the FU
  bufRef->release();
}


//______________________________________________________________________________
inline
void BU::buCollectMsg(toolbox::mem::Reference *bufRef)
{
  LOG4CPLUS_FATAL(this->getApplicationLogger(), "buCollectMsg() NOT IMPLEMENTED");
  exit(-1);
}


//______________________________________________________________________________
inline
void BU::buDiscardNMsg(toolbox::mem::Reference *bufRef)
{
  // Does nothing but free the incoming I2O message
  nbEventsDiscarded_.value_++;
  bufRef->release();
}


//______________________________________________________________________________
inline
void BU::generateNFedFragments(double     fedSizeMean,
			       double     fedSizeWidth,
			       uintvec_t& fedSize)
{
  for(unsigned int i=0;i<fedSize.size();i++) {
    int iFedSize(0);
    while (iFedSize<(fedTrailerSize_+fedHeaderSize_)) {
      double logSize=RandGauss::shoot(std::log(fedSizeMean),
				      std::log(fedSizeMean)-
				      std::log(fedSizeWidth/2.));
      iFedSize=(int)(std::exp(logSize));
      
      iFedSize-=iFedSize % 8; // all blocks aligned to 64 bit words
    }
    fedSize[i]=iFedSize;
  }
}


#endif
