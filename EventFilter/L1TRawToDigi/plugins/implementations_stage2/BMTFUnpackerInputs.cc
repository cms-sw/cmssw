#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "BMTFUnpackerInputs.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

namespace l1t
{
  namespace stage2
  {
    void numWheelSectorTrTag_bmtf(int& wheelNo, int& tagSegID, int linkNo, int amcNo)
    {
      if (linkNo >= 0 && linkNo < 6)
	wheelNo = -2;
      else if (linkNo >= 8 && linkNo < 14)
	wheelNo = -1;
      else if (linkNo >= 16 && linkNo < 22)
	wheelNo = 0;	
      else if (linkNo >= 22 && linkNo < 28)
	wheelNo = 1;
      else if ( (linkNo >= 28 && linkNo < 30) || (linkNo >= 32 && linkNo < 36))
	wheelNo = 2;
			
      if ( linkNo%2 == 0 )
	tagSegID = 0;
      else
	tagSegID = 1;
    }

    bool checkQual_bmtf(const unsigned int& value, const bool& isNewFw)
    {
      if (isNewFw)
	return (value == 7);
      else 
	return (value == 0);
    }

    bool unpacking_bmtf(const Block& block, UnpackerCollections *coll, qualityHits& linkAndQual_, const bool& isNewFw)
    {

      unsigned int ownLinks[] = {4,5,12,13,20,21,22,23,28,29};
      bool ownFlag(false);

      //Checks if the given block coresponds to 1 of the OWN links
      for (int i = 0; i < 10; i++)
	{
	  if (block.header().getID()/2 == ownLinks[i])
	    ownFlag = true;
	}
      if ( !ownFlag )//if not returns that the "job here done"
	return true;


      //Get header ID and payload from the given Block
      unsigned int blockId = block.header().getID();
      LogDebug("L1T") << "Block ID: " << blockId << " size: " << block.header().getSize();
      auto payload = block.payload();
			
      //Make output CMSSW collections
      L1MuDTChambPhContainer *resPhi = static_cast<BMTFCollections*>(coll)->getInMuonsPh();
      L1MuDTChambThContainer *resThe = static_cast<BMTFCollections*>(coll)->getInMuonsTh();

      //Get input phi & eta Containers
      L1MuDTChambPhContainer::Phi_Container phiData = *(resPhi->getContainer()); 
      L1MuDTChambThContainer::The_Container theData = *(resThe->getContainer());
			
      //ZeroSuppresion Handler
      BxBlocks bxBlocks;
      bool ZS_enabled = (bool)((block.header().getFlags() >> 1) & 0x01);//getFlags() returns first 8-bits from the amc header
      if (ZS_enabled)
	bxBlocks = block.getBxBlocks((unsigned int)6, true);//it returnes 7-32bit bxBlocks originated from the amc13 Block
      else
	bxBlocks = block.getBxBlocks((unsigned int)6, false);//it returnes 6-32bit bxBlocks originated from the amc13 Block


      for(auto ibx : bxBlocks)//Bx iteration
	{

	  int bxNum = ibx.header().getBx();
	  uint32_t inputWords[ibx.getSize()]; //array of 6 uint32_t payload-words (size of the payload in the BxBlock)

	  //Note
	  /*In the non-ZS fashion, the expression "block.header().getSize()/nBX" was 6 in any case
	    the reason is that the size is 6 or 30, and these numbers are divided by 1 or 5 respectively.*/

	  //Fill the above uint32_t array
	  for(unsigned int iw = 0; iw < ibx.getSize(); iw++)
	    inputWords[iw] = (ibx.payload())[iw];

			
	  int wheel, sector, trTag;//Container information
	  numWheelSectorTrTag_bmtf(wheel, trTag, blockId/2, block.amc().getAMCNumber());//this returns wheel & tsTag
	  sector = block.amc().getBoardID() - 1;

	  //Check if the sector is "out of range" - (trys then to use AMC13 information?)
	  if ( sector < 0 || sector > 11 )
	    {
	      edm::LogInfo ("l1t:stage2::BMTFUnpackerInputs::unpack") << "Sector found out of range so it will be calculated by the slot number";
	      if ( block.amc().getAMCNumber()%2 != 0 )
		sector = block.amc().getAMCNumber()/2 ;
	      else
		sector = 6 + (block.amc().getAMCNumber()/2 -1);
	    }

	  int mbPhi[4], mbPhiB[4], mbQual[4], mbBxC[4], mbRPC[4];//Container information
	  //mbPhiB[2] = 0;
				
	  for (int iw = 0; iw < 4; iw++)// 4 phi (32-bit) words
	    {
	      if ( ((inputWords[iw] & 0xfffffff) == 0) || (inputWords[iw] == 0x505050bc) ) 
		continue;
	      else if ( (inputWords[iw] != 0x505050bc) && (inputWords[iw+2] == 0x505050bc) )
		continue;
						
					
	      if ( ((inputWords[iw] >> 11) & 0x1) == 1 )
		mbPhi[iw] = (inputWords[iw] & 0x7FF ) - 2048;
	      else
		mbPhi[iw] = (inputWords[iw] & 0xFFF );
						
				
	      if ( ((inputWords[iw] >> 21) & 0x1) == 1 )
		mbPhiB[iw] = ( (inputWords[iw] >> 12) & 0x1FF ) - 512;
	      else
		mbPhiB[iw] = (inputWords[iw] >> 12) & 0x3FF;
					
	      mbQual[iw] = (inputWords[iw] >> 22) & 0x7;
	      mbRPC[iw] = (inputWords[iw] >> 26) & 0x1;
	      mbBxC[iw] = (inputWords[iw] >> 30) & 0x3;

	      //if (mbQual[iw] == 0)
	      if (checkQual_bmtf(mbQual[iw], isNewFw))
		continue;
					
	      phiData.push_back( L1MuDTChambPhDigi( bxNum, wheel, sector, iw+1, mbPhi[iw], mbPhiB[iw], mbQual[iw], trTag, mbBxC[iw], mbRPC[iw] ) );

	    }//4 phi words
				
				
	  int etaHits[3][7];//Container information
	  bool zeroFlag[3];
	  for (int i = 0; i < 3; i++)// 3 eta (7-bit) words
	    {
	      zeroFlag[i] = false;
	      for(int j=0; j<7; j++)
		{
		  etaHits[i][j] = (inputWords[4] >> (i*7 + j)) & 0x1;
		  if ( etaHits[i][j]!=0 )
		    zeroFlag[i] = true;
		}
	    }//3 eta words


	  if ( trTag == 1 )
	    {
	      for (int i = 0; i < 3; i++)
		{
		  if (zeroFlag[i])
		    theData.push_back(L1MuDTChambThDigi( bxNum, wheel, sector, i+1, etaHits[i], linkAndQual_.hits[i]) );
		}

	    }
	  else
	    {
	      /*
		qualityHits temp;
		temp.linkNo = blockId/2;
		std::copy(&etaHits[0][0], &etaHits[0][0]+3*7,&temp.hits[0][0]);
		linkAndQual_[blockId/2] = temp;	
	      */
	      linkAndQual_.linkNo = blockId/2;
	      std::copy(&etaHits[0][0], &etaHits[0][0]+3*7,&linkAndQual_.hits[0][0]);
	    }


	}//iBxBlock

      //Fill Containers
      resThe->setContainer(theData);
      resPhi->setContainer(phiData);

      return true;
			
    }
		
    bool BMTFUnpackerInputsOldQual::unpack(const Block& block, UnpackerCollections *coll)
    {
      return unpacking_bmtf(block, coll, linkAndQual_, false);
    }//unpack old quality

    bool BMTFUnpackerInputsNewQual::unpack(const Block& block, UnpackerCollections *coll)
    {
      return unpacking_bmtf(block, coll, linkAndQual_, true);
    }//unpack new quality
  }//ns2
}//ns l1t;

DEFINE_L1T_UNPACKER(l1t::stage2::BMTFUnpackerInputsOldQual);
DEFINE_L1T_UNPACKER(l1t::stage2::BMTFUnpackerInputsNewQual);
