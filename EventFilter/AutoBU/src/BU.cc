////////////////////////////////////////////////////////////////////////////////
//
// BU
// --
//
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/AutoBU/interface/BU.h"
#include "EventFilter/Utilities/interface/Crc.h"


#include <netinet/in.h>


using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
BU::BU(xdaq::ApplicationStub *s) 
  : xdaq::WebApplication(s)
  , dataBufSize_(4096)
  , nSuperFrag_(64)
  , nbEventsSent_(0)
  , nbEventsDiscarded_(0)
  , fedSizeMean_(1024)    // mean  of fed size for rnd generation
  , fedSizeWidth_(1024)   // width of fed size for rnd generation
  , useFixedFedSize_(false)
  , pool_(0)
{
  xdata::InfoSpace *is = getApplicationInfoSpace();
  
  is->fireItemAvailable("dataBufSize",      &dataBufSize_);
  is->fireItemAvailable("nSuperFrag",       &nSuperFrag_);
  is->fireItemAvailable("nbEventsSent",     &nbEventsSent_);
  is->fireItemAvailable("nbEventsDiscarded",&nbEventsDiscarded_);
  is->fireItemAvailable("fedSizeMean",      &fedSizeMean_);
  is->fireItemAvailable("fedSizeWidth",     &fedSizeWidth_);
  is->fireItemAvailable("useFixedFedSize",  &useFixedFedSize_);
  
  i2o::bind(this,
	    &BU::buAllocateNMsg,
	    I2O_BU_ALLOCATE,
	    XDAQ_ORGANIZATION_ID);
  
  i2o::bind(this,
	    &BU::buCollectMsg,
	    I2O_BU_COLLECT,
	    XDAQ_ORGANIZATION_ID);
  
  i2o::bind(this,
	    &BU::buDiscardNMsg,
	    I2O_BU_DISCARD,
	    XDAQ_ORGANIZATION_ID);
  
  try {
    toolbox::mem::HeapAllocator *allocator=new toolbox::mem::HeapAllocator();
    toolbox::net::URN urn("toolbox-mem-pool","ABU");
    toolbox::mem::MemoryPoolFactory* poolFactory=
      toolbox::mem::getMemoryPoolFactory();
    
    pool_=poolFactory->createPool(urn, allocator);
    
    LOG4CPLUS_INFO(getApplicationLogger(),"Created memory pool: "<<"ABU");
  }
  catch (toolbox::mem::exception::Exception& e) {
    string s="Failed to create pool: ABU ";
    LOG4CPLUS_FATAL(getApplicationLogger(),s);
    XCEPT_RETHROW(xcept::Exception,s,e);
  }
}


//______________________________________________________________________________
BU::~BU()
{
  
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
toolbox::mem::Reference *BU::createSuperFrag(const I2O_TID& fuTid,
					     const U32&     fuTransaction,
					     const U32&     trigNo,
					     const U32&     iSuperFrag,
					     const U32&     nSuperFrag)
{
  bool   errorFound(false);
  bool   configFeds=(0==PlaybackRawDataProvider::instance());
  
  size_t msgHeaderSize   =sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
  size_t fullBlockPayload=dataBufSize_-msgHeaderSize;
  
  if((fullBlockPayload%4)!=0) {
    LOG4CPLUS_FATAL(getApplicationLogger(),"The full block payload of "
		    <<fullBlockPayload<<" bytes is not a multiple of 4");
    errorFound = true;
  }
  
  unsigned int nBlock=estimateNBlocks(fedSize_,fullBlockPayload);

  if(nBlock==0) {
    LOG4CPLUS_FATAL(getApplicationLogger(),"No blocks to be created for the chain");
    errorFound = true;
  }
  
  if(errorFound) exit(-1);
  
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
      bufRef=toolbox::mem::getMemoryPoolFactory()->getFrame(pool_,dataBufSize_);
    }
    catch(...) {
      LOG4CPLUS_FATAL(getApplicationLogger(),"xdaq::frameAlloc failed");
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
      if((iFed==(fedSize_.size()-1)) && !last) {
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
	if(iFed==fedSize_.size()-1) {
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
      while(iFed<fedSize_.size()) {
	
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
	
      } // while (iFed<fedSize_.size())
      
      // earmark the last block
      if (iFed==fedSize_.size() && remainder==0 && !last) {
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
int BU::estimateNBlocks(const uintvec_t& fedSize,size_t fullBlockPayload)
{
  int result(0);
  
  U32 curbSize=frlHeaderSize_;
  U32 totSize =curbSize;
  
  for(unsigned int i=0;i<fedSize.size();i++) {
    curbSize+=fedSize[i];//+fedHeaderSize_+fedTrailerSize_;
    totSize +=fedSize[i];//+fedHeaderSize_+fedTrailerSize_;
    
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
  
  //LOG4CPLUS_ERROR(adapter_->getApplicationLogger(),
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
    
    //LOG4CPLUS_ERROR(adapter_->getApplicationLogger(),
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
