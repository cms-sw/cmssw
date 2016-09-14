#include "L1Trigger/L1TMuon/interface/RegionalMuonRawDigiTranslator.h"

#include "BMTFUnpackerOutput.h"

namespace l1t
{
	namespace stage2
	{
		bool BMTFUnpackerOutput::unpack(const Block& block, UnpackerCollections *coll)
		{
			unsigned int blockId = block.header().getID();
			LogDebug("L1T") << "Block ID: " << blockId << " size: " << block.header().getSize();
			
			auto payload = block.payload();
			
			//int nwords(2); //two words per muon
			int nBX, firstBX, lastBX;
			nBX = int(ceil(block.header().getSize()/6));
			
			getBXRange(nBX, firstBX, lastBX);
			//if we want to use central BX, uncommect the two lines below
			//firstBX=0;
			//lastBX=0;
			//LogDebug("L1T") << "BX override. Set firstBX = lastBX = 0";
			
			RegionalMuonCandBxCollection *res;
			res = static_cast<BMTFCollections*>(coll)->getBMTFMuons();
			res->setBXRange(firstBX, lastBX);
			
			LogDebug("L1T") << "nBX = " << nBX << " firstBX = " << firstBX << " lastBX = " << lastBX;
			
			int processor = block.amc().getBoardID() - 1;
			if ( processor < 0 || processor > 11 )
			{
				edm::LogInfo ("l1t:stage2::BMTFUnpackerOutput::unpack") << "Processor found out of range so it will be calculated by the old way";
				if ( block.amc().getAMCNumber()%2 != 0 )
			                processor = block.amc().getAMCNumber()/2 ;
			        else
			                processor = 6 + (block.amc().getAMCNumber()/2 -1);
			}

			for(int ibx = firstBX; ibx <= lastBX; ibx++)
			{
				int ip(0);
				for(unsigned int iw = 0; iw < block.header().getSize()/nBX; iw += 2)
				{
					uint32_t raw_first = payload[ip+(ibx+lastBX)*6];
					ip++;
					uint32_t raw_secnd = payload[ip+(ibx+lastBX)*6];
					ip++;
					if ( raw_first == 0 )
					{
						LogDebug("L1T") << "Raw data is zero";
						continue;
					}
					
					RegionalMuonCand muCand = RegionalMuonCand();
					RegionalMuonRawDigiTranslator::fillRegionalMuonCand(muCand, raw_first, raw_secnd, processor, tftype::bmtf);
					muCand.setLink(48 + processor);	//the link corresponds to the uGMT input

					LogDebug("L1T") << "Pt = " << muCand.hwPt() << " eta: " << muCand.hwEta() << " phi: " << muCand.hwPhi();
					if ( muCand.hwQual() != 0 )
					{
						res->push_back(ibx, muCand);
					}
					
				}//for iw
			}//for ibx

			return true;
		}//unpack
	}//ns stage2
}//ns lt1
			
