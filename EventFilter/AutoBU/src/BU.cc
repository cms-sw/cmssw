#include "EventFilter/AutoBU/interface/BU.h"

#include <netinet/in.h>

    /**
     * Creates and returns a chain of blocks/messages representing a dummy
     * super fragment.
     */

toolbox::mem::Reference *BU::createSuperFrag
(const size_t  dataBufSize,
 const U32     fragSize,
 const I2O_TID fuTid,
 const U32     eventHandle,
 const U32     fuTransaction,
 const U8      currentFragment,
 const U8      totalFragments
 // DaqEvent *ev_
)
  
{

  size_t frlhs = sizeof(frlh_t);
  size_t fedhs = sizeof(fedh_t);
  size_t fedts = sizeof(fedt_t);
  size_t msgHeaderSize       = sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
  size_t fullBlockPayload    = dataBufSize - msgHeaderSize;

  vector<int> sfed(16); // assume for now 2 FEDs/FRL, 8 FRL/RU
  vector<char*> data(16);
  //  if(ev_==0)
    {
      if(useFixedSize_)
	sfed.assign(sfed.size(),fragSize);
      else
	generateNFedFragments(float(fragSize), float(fragSize), 
			      sfed);
    }
    /*
  else
    {
      for(unsigned i = 0; i < 16; i++)
	{
	  DaqFEDRawData *dfr = ev_->findFEDRawData((currentFragment*16)+i);
	  
	  if(dfr && (dfr->size()!=0))
	    {
	      sfed[i] = dfr->size();
	      if(sfed[i]%8 != 0)
		cerr << "FED Format is NOT padded to 64 bits ! size:" 
		     << sfed[i] << endl;
	      data[i] = dfr->data();
	    }
	  else
	    {
	      sfed[i] = 0;
	      data[i] = 0;
	    }
	}
    }
    */

  U8     totalBlocks         = estimateNBlocks(sfed,fullBlockPayload);
  U8     currentBlock        = 0;
  bool   errorFound          = false;
  U32    payload             = 0;
  toolbox::mem::Reference      *head               = 0;
  toolbox::mem::Reference      *tail               = 0;
  toolbox::mem::Reference      *bufRef             = 0;
  I2O_MESSAGE_FRAME            *stdMsg             = 0;
  I2O_PRIVATE_MESSAGE_FRAME    *pvtMsg             = 0;
  I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block        = 0;
  unsigned char *startOfPayload                    = 0;
	

  if(totalBlocks == 0)
    {
      LOG4CPLUS_FATAL(this->getApplicationLogger(),
		      "No blocks to be created for the chain");

      errorFound = true;
    }

  if((fullBlockPayload % 4) != 0)
    {
      LOG4CPLUS_FATAL(this->getApplicationLogger(),
		      "The full block payload of " << fullBlockPayload
		      << " bytes is not a multiple of 4");

      errorFound = true;
    }


  if(errorFound)
    {
      exit(-1);
    }

  unsigned int fedind=0;
  int remainder = 0;
  int segment = 0;
  bool last = false;
  bool warning = false;
  int totSize = 0;
  for(currentBlock=0; currentBlock<totalBlocks; currentBlock++)
    {
      // If last block and its partial (there can be only 0 or 1 partial)
      payload = fullBlockPayload;
      //	  cout << "At block " << (int)currentBlock << endl;

      // Allocate memory for a fragment block / message
      try
	{
	  bufRef = 
	    toolbox::mem::getMemoryPoolFactory()->getFrame(pool_,dataBufSize);
	}
      catch(...)
	{
	  LOG4CPLUS_FATAL(this->getApplicationLogger(), "xdaq::frameAlloc failed");

	  exit(-1);
	}

      // Fill in the fields of the fragment block / message
      stdMsg = (I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
      pvtMsg = (I2O_PRIVATE_MESSAGE_FRAME*)stdMsg;
      block  = (I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)stdMsg;

      memset(block, 0, sizeof(I2O_MESSAGE_FRAME));

      pvtMsg->XFunctionCode    = I2O_FU_TAKE;
      pvtMsg->OrganizationID   = XDAQ_ORGANIZATION_ID;

      stdMsg->MessageSize      = (msgHeaderSize + payload) >> 2;
      stdMsg->Function         = I2O_PRIVATE_MESSAGE;
      stdMsg->VersionOffset    = 0;
      stdMsg->MsgFlags         = 0;  // Point-to-point
      stdMsg->InitiatorAddress = i2o::utils::getAddressMap()->getTid(getApplicationDescriptor());
      stdMsg->TargetAddress    = fuTid;

      block->buResourceId                  = eventHandle;
      block->fuTransactionId               = fuTransaction;
      block->blockNb                       = currentBlock;
      block->nbBlocksInSuperFragment       = totalBlocks;
      block->superFragmentNb               = currentFragment;
      block->nbSuperFragmentsInEvent       = totalFragments;

      // Fill in payload 

      startOfPayload          = (unsigned char*)block + msgHeaderSize;
      frlh_t* frlh = (frlh_t*)startOfPayload;
      frlh->trigno = /*htonl(*/eventHandle/*)*/;
      frlh->segno  = /*htonl(*/segment    /*)*/;
      segment++;
      unsigned char *startOfFedBlocks = startOfPayload + frlhs;
      payload -= frlhs;
      frlh->segsize = payload;
      int leftspace = payload;

      // a fed trailer was left over from the previous block
      if(remainder == -1)
	{
	  fedt_t *fedt = (fedt_t*)startOfFedBlocks;
	  fedt->eventsize = sfed[fedind]+fedhs+fedts;
	  fedt->eventsize /= 8; //wc in fed trailer in 64bit words
	  fedt->eventsize |= 0xa0000000;
	  fedt->conscheck = 0x0;
	  startOfFedBlocks += fedts;
	  leftspace -= fedts;
	  remainder = 0;
	  // make sure this is not the end, if it is, earmark
	  if((fedind == (sfed.size()-1)) && !last)
	    {
	      frlh->segsize -= leftspace;
	      int orsize = stdMsg->MessageSize <<2;
	      orsize -= leftspace;
	      bufRef->setDataSize(orsize);
	      stdMsg->MessageSize = orsize >> 2;
	      frlh->segsize = frlh->segsize | FRL_LAST_SEGM;
	      last = true;
	    }
	  fedind++; // must increment the fed index
	}
      // the remainder of the last fed fits with its trailer
      if(remainder >0 && payload >= (remainder+fedts))
	{
	  if(data[fedind]!=0)
	    memcpy(startOfFedBlocks,data[fedind]+sfed[fedind]-remainder,remainder);
	  startOfFedBlocks += remainder;
	  leftspace -= remainder;
	  fedt_t *fedt = (fedt_t*)startOfFedBlocks;
	  fedt->eventsize = sfed[fedind]+fedhs+fedts;
	  fedt->eventsize /= 8; //wc in fed trailer in 64bit words
	  fedt->eventsize |= 0xa0000000;
	  fedt->conscheck = 0x0;
	  startOfFedBlocks += fedts;
	  leftspace -= fedts;
		
	  // if this is the last segment, earmark it
	  if(fedind == sfed.size()-1)
	    {
	      frlh->segsize -= leftspace;
	      int orsize = stdMsg->MessageSize <<2;
	      orsize -= leftspace;
	      bufRef->setDataSize(orsize);
	      stdMsg->MessageSize = orsize >> 2;
	      frlh->segsize = frlh->segsize | FRL_LAST_SEGM;
	      last = true;
	    }
	  remainder = 0;
	  fedind++; // must increment the fed index
	}
      // the remainder of the previous fed fits but not its trailer
      else if(remainder >0 && payload >= remainder && 
	      payload < (remainder+fedts))
	{
	  if(data[fedind]!=0)
	    memcpy(startOfFedBlocks,data[fedind]+sfed[fedind]-remainder,remainder);
	  frlh->segsize = remainder;
	  remainder = -1;
	  leftspace = 0;
	}
      // the remainder does not fit, the block is completely filled 
      else if(remainder >0)
	{
	  if(data[fedind]!=0)
	    memcpy(startOfFedBlocks,data[fedind]+sfed[fedind]-remainder,payload);
	  remainder -= payload;
	  leftspace = 0;
	}
      // no remainder is left
      if(remainder == 0)
	{
	  // loop on feds
	  while(fedind < sfed.size())
	    {
	      // if the next header does not fit, jump to following block
	      if(leftspace < fedhs)
		{
		  frlh->segsize -= leftspace;
		  break;
		}
	      // fill in header data
	      fedh_t *fedh = (fedh_t*)startOfFedBlocks;
	      fedh->eventid = eventHandle;
	      fedh->eventid |= 0x50000000;
	      fedh->sourceid = ((fedind+currentFragment*16) << 8) & 
		FED_SOID_MASK;

	      leftspace -= fedhs;
	      startOfFedBlocks += fedhs;
	      // fed fits with its trailer
	      if(sfed[fedind] + fedts <= leftspace)
		{
		  if(data[fedind]!=0)
		    memcpy(startOfFedBlocks,data[fedind],sfed[fedind]);
		  startOfFedBlocks += sfed[fedind];
		  fedt_t *fedt = (fedt_t*)startOfFedBlocks;
		  fedt->eventsize = sfed[fedind]+fedhs+fedts;
		  fedt->eventsize /= 8; //wc in fed trailer in 64bit words
		  fedt->eventsize |= 0xa0000000;
		  fedt->conscheck = 0x0;
		  leftspace -= sfed[fedind] + fedts;
		  startOfFedBlocks += fedts;
		}
	      // fed fits but not trailer
	      else if(sfed[fedind] <= leftspace)
		{
		  if(data[fedind]!=0)
		    memcpy(startOfFedBlocks,data[fedind],sfed[fedind]);
		  leftspace -= sfed[fedind];
		  frlh->segsize -= leftspace;
		  remainder = -1;
		  break;
		}
	      // fed does not fit
	      else if(sfed[fedind]>leftspace)
		{
		  if(data[fedind]!=0)
		    memcpy(startOfFedBlocks,data[fedind],leftspace);
		  remainder = sfed[fedind]-leftspace;
		  leftspace = 0;
		  break;
		}
	      fedind++;
	    } // close loop on fed indices

	  // earmark the last block
	  if(fedind == sfed.size() && remainder == 0 && !last)
	    {
	      frlh->segsize -= leftspace;
	      int orsize = stdMsg->MessageSize <<2;
	      orsize -= leftspace;
	      bufRef->setDataSize(orsize);
	      stdMsg->MessageSize = orsize >> 2;
	      frlh->segsize = frlh->segsize | FRL_LAST_SEGM;
	      last = true;
	    }
		
	}
      payloadNb_++;
      totSize += (frlh->segsize & FRL_SEGSIZE_MASK) + frlhs;
      if(currentBlock == 0) // This is the first fragment block / message
	{
	  head = bufRef;
	  tail = bufRef;
	}
      else
	{
	  tail->setNextReference(bufRef); //set link in list
	  tail = bufRef;
	}
      if((currentBlock==totalBlocks-1) && remainder !=0)
	{
	  totalBlocks +=1;
	  warning = true;
	}
    }
  // fix case where block estimate was wrong
  if(warning)
    {
      bufRef = head;
      do{
	stdMsg = (I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
	pvtMsg = (I2O_PRIVATE_MESSAGE_FRAME*)stdMsg;
	block  = (I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)stdMsg;
	block->nbBlocksInSuperFragment       = totalBlocks;		
      }while((bufRef = bufRef->getNextReference()));
    }
  //	cout << "Finished formatting " << endl;
  return head; // return the top of the chain
}


int BU::estimateNBlocks(vector<int> &sfed, size_t fullBlockPayload)
{
  size_t frlhs = sizeof(frlh_t);
  size_t fedhs = sizeof(fedh_t);
  size_t fedts = sizeof(fedt_t);

  U32 curbSize = frlhs;
  U32 totSize = curbSize;
  int nBlocks = 0;
  for(unsigned int i = 0; i<sfed.size(); i++) 
    {
      curbSize += sfed[i] + fedhs + fedts;
      totSize +=  sfed[i] + fedhs + fedts;
      // the calculation of the number of blocks needed
      // must 
      // handle the fact that RU can accommodate more than one frl blocks
      // and remove intermediate headers
      
      if(curbSize > fullBlockPayload)
	{
	  curbSize += frlhs * (curbSize/fullBlockPayload);
	  nBlocks += curbSize/fullBlockPayload;
	  if(curbSize%fullBlockPayload>0)
	    totSize += frlhs * (curbSize/fullBlockPayload);
	  else
	    totSize += frlhs * ((curbSize/fullBlockPayload)-1);
	  curbSize = curbSize%fullBlockPayload;
	}
    }	
  if(curbSize != 0) nBlocks++;
  // int estb = nBlocks;
  nBlocks = totSize/fullBlockPayload + (totSize%fullBlockPayload>0 ? 1 : 0);
  /*      if(estb!=nBlocks) cout << "Warning:: estb != nBlocks " 
	  << estb << " " <<  nBlocks << endl;*/
  
  return nBlocks;
}


int BU::check_event_data(unsigned long blocks_adrs[], int nmb_blocks/*,
								      DaqEvent *ev*/)
{
  int retval = 0;
  int   feds = -1; // fed size
  int   fedid = -1;
  int fedhs = sizeof(fedh_t); // for convenience, store the header size
  int fedts = sizeof(fedt_t); // for convenience, store the trailer size
  int frlhs = sizeof(frlh_t); // for convenience, store the trailer size
  char * fedd = 0; //temporary buffer for fed data

  unsigned char * blk_cursor = 0;

  int current_trigno = -1;

  int seglen_left = 0;
  int fed_left = 0;
  
  //loop on blocks starting from last
  for(int iblk = nmb_blocks-1; iblk>=0; iblk--)
    {
      blk_cursor = (unsigned char*)blocks_adrs[iblk];
      frlh_t *ph = (frlh_t *)blk_cursor;

      int hd_trigno  = ph->trigno;
      // int hd_segno   = ph->segno;
      int hd_segsize     = ph->segsize;
      int segsize_proper = hd_segsize & ~FRL_LAST_SEGM ;

      // check trigno
      if (current_trigno == -1) {
	current_trigno = hd_trigno ;
      }
      else {
	if (current_trigno != hd_trigno) {
	  printf("data error nmb_blocks %d iblock %d trigno expect %d got %d \n"
		 ,nmb_blocks,iblk,current_trigno,hd_trigno) ;
	  return (-1) ;
	  
	} 
      }
    
      // check that last block flagged as last segment and none of the others
      if (iblk == nmb_blocks-1) {
	if  (!(hd_segsize & FRL_LAST_SEGM)) {
	  printf("data error nmb_blocks %d iblock %d last_segm not set \n",nmb_blocks,iblk) ;
	  
	  return (-1) ;
	}
      }
      else {
	if  ((hd_segsize & FRL_LAST_SEGM)) {
	  printf("data error nmb_blocks %d iblock %d last_segm  set \n",nmb_blocks,iblk) ;
	  return (-1) ;
	}
      }
    


      blk_cursor += frlhs;
      seglen_left = segsize_proper;
      blk_cursor += segsize_proper;
      while(seglen_left>=0)
	{
	  if(fed_left == 0)
	    {
	      if(feds>=0) {
		retval += 0;/*check_fed(fedd,feds,fedid,ev);*/ //will check consistency with input event 
		delete[] fedd;
		feds = -1;
	      }

	      if(seglen_left==0)break;
	      seglen_left -= fedts;
	      blk_cursor -= fedts;
	      fedt_t *pft = (fedt_t*)blk_cursor;
	      int fedlen = pft->eventsize & FED_EVSZ_MASK;   // 24 bits  - len in bytes fed payload 

	      fedlen *=8; // in the fed trailer, wc is in 64 bit words
	      
	      feds = fedlen-fedhs-fedts;
	      // have the fed size, now reserve the buffer
	      fedd = new char[feds];
	      if((seglen_left-(fedlen-fedts)) >= 0)
	      {
		blk_cursor -= feds;
		memcpy(fedd,blk_cursor,feds);
		seglen_left -= (fedlen-fedts);
		fed_left = 0;
		blk_cursor -= fedhs;
		fedh_t *pfh = (fedh_t *)blk_cursor;
		fedid = pfh->sourceid & FED_SOID_MASK;
		fedid = fedid >> 8;
	      }
	      else
	      {
		blk_cursor = (unsigned char*)blocks_adrs[iblk]+frlhs;
		fed_left = fedlen-fedts-seglen_left;
		memcpy(fedd+feds-seglen_left,blk_cursor,seglen_left);
		seglen_left = 0;
	      }
	    }
	  else if(fed_left > fedhs)
	    {
	      if(seglen_left==0)break;
	      if(seglen_left - fed_left >= 0)
	      {
		blk_cursor -= (fed_left-fedhs);
		memcpy(fedd,blk_cursor,fed_left-fedhs);
		seglen_left -= fed_left;
		blk_cursor -= fedhs;
		fed_left = 0;
		fedh_t *pfh = (fedh_t *)blk_cursor;
		fedid = pfh->sourceid & FED_SOID_MASK;
		fedid = fedid >> 8;
	      }
	      else if(seglen_left - fed_left + fedhs > 0)
	      {
		blk_cursor = (unsigned char*)blocks_adrs[iblk]+frlhs;
		memcpy(fedd,blk_cursor,fed_left-fedhs);
		fed_left = fedhs;
		seglen_left = 0;
	      }
	      else
	      {
		blk_cursor = (unsigned char*)blocks_adrs[iblk]+frlhs;
		memcpy(fedd+fed_left-fedhs-seglen_left,blk_cursor,seglen_left);
		fed_left -= seglen_left;
		seglen_left = 0;
	      }
	    }
	  else
	  {
	      if(seglen_left==0)break;
	      blk_cursor -= fedhs;
	      fed_left = 0;
	      seglen_left -= fedhs;
	      fedh_t *pfh = (fedh_t *)blk_cursor;
	      fedid = pfh->sourceid & FED_SOID_MASK;
	      fedid = fedid >> 8;
	    }
	}
      //      dumpFrame((unsigned char*)blocks_adrs[iblk],segsize_proper+frlhs);
    }
  return retval;
}
/*
int BU::check_fed(char *fedd, int feds, int fedid, DaqEvent *ev)
{
  DaqFEDRawData *dfr = ev->findFEDRawData(fedid);  
  if(dfr)
    return compareFrame(fedd,feds,dfr->data(),dfr->size());
  else
    {
      return -1;
    }
}
*/


XDAQ_INSTANTIATOR_IMPL(BU)
