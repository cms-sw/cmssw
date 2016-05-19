#include "BMTFUnpackerInputs.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

namespace l1t
{
	namespace stage2
	{
		void numWheelSectorTrTag(int& wheelNo, int& tagSegID, int linkNo, int amcNo)
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
			
		bool BMTFUnpackerInputs::unpack(const Block& block, UnpackerCollections *coll)
		{
			unsigned int ownLinks[] = {4,5,12,13,20,21,22,23,28,29};
			bool ownFlag(false);
			for (int i = 0; i < 10; i++)
			{
				if (block.header().getID()/2 == ownLinks[i])
					ownFlag = true;
			}
			if ( !ownFlag )
				return true;

			unsigned int blockId = block.header().getID();
			LogDebug("L1T") << "Block ID: " << blockId << " size: " << block.header().getSize();
			auto payload = block.payload();
			int nBX, firstBX, lastBX;
		
			nBX = int(ceil(block.header().getSize()/6));
			getBXRange(nBX, firstBX, lastBX);

			LogDebug("L1T") << "BX override. Set firstBX = lastBX = 0";
			
			L1MuDTChambPhContainer *resPhi = static_cast<BMTFCollections*>(coll)->getInMuonsPh();
			L1MuDTChambThContainer *resThe = static_cast<BMTFCollections*>(coll)->getInMuonsTh();

			L1MuDTChambPhContainer::Phi_Container phiData = *(resPhi->getContainer()); 
			L1MuDTChambThContainer::The_Container theData = *(resThe->getContainer());
			
			
			for(int ibx = firstBX; ibx <= lastBX; ibx++)
			{
				uint32_t inputWords[block.header().getSize()/nBX];
				
				for(unsigned int iw = 0; iw < block.header().getSize()/nBX; iw++)
					inputWords[iw] = payload[iw+(ibx+lastBX)*6];
			
				int wheel, sector, trTag;
				numWheelSectorTrTag(wheel, trTag, blockId/2, block.amc().getAMCNumber());
				sector = block.amc().getBoardID() - 1;
				if ( sector < 0 || sector > 11 )
				{
					edm::LogInfo ("l1t:stage2::BMTFUnpackerInputs::unpack") << "Sector found out of range so it will be calculated by the old way";
					if ( block.amc().getAMCNumber()%2 != 0 )
				                sector = block.amc().getAMCNumber()/2 ;
				        else
				                sector = 6 + (block.amc().getAMCNumber()/2 -1);
				}

				int mbPhi[4], mbPhiB[4], mbQual[4], mbBxC[4], mbRPC[4];
				//mbPhiB[2] = 0;
				
				for (int iw = 0; iw < 4; iw++)
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
					
					mbQual[iw] = (inputWords[iw] >> 22) & 0xF;
					mbRPC[iw] = (inputWords[iw] >> 26) & 0x1;
					mbBxC[iw] = (inputWords[iw] >> 30) & 0x3;
					if (mbQual[iw] == 0)
						continue;
					
					phiData.push_back( L1MuDTChambPhDigi( ibx, wheel, sector, iw+1, mbPhi[iw], mbPhiB[iw], mbQual[iw], trTag, mbBxC[iw], mbRPC[iw] ) );
				}//iw
				
				
				int etaHits[3][7];//, etaHitsBxC;
				bool zeroFlag[3];
				for (int i = 0; i < 3; i++)
				{
					zeroFlag[i] = false;
					for(int j=0; j<7; j++)
					{
						etaHits[i][6-j] = (inputWords[4] >> (i*7 + j)) & 0x1;
						if ( etaHits[i][6-j]!=0 )
							zeroFlag[i] = true;
					}
				}
				if ( trTag == 1 )
				{
					for (int i = 0; i < 3; i++)
					{
						if (zeroFlag[i])
							theData.push_back(L1MuDTChambThDigi( ibx, wheel, sector, i+1, etaHits[i], linkAndQual_[blockId/2 - 1].hits[i]) );
					}

				}
				else
				{
					qualityHits temp;
					temp.linkNo = blockId/2;
					std::copy(&etaHits[0][0], &etaHits[0][0]+3*7,&temp.hits[0][0]);
					linkAndQual_[blockId/2] = temp;	
				}

			}//ibx
			resThe->setContainer(theData);
			resPhi->setContainer(phiData);
			
			
		return true;
		}//unpack
	}//ns2
}//ns l1t;
