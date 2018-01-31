#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "L1Trigger/L1TMuon/interface/RegionalMuonRawDigiTranslator.h"

#include "BMTFUnpackerOutput.h"
//#include "testTools.h"//debug

namespace l1t
{
  namespace stage2
  {
    bool BMTF_ZSUnpackerOutput::unpack(const Block& block, UnpackerCollections *coll)
    {
      unsigned int blockId = block.header().getID();
      LogDebug("L1T") << "Block ID: " << blockId << " size: " << block.header().getSize();
			
      auto payload = block.payload();

      //ZeroSupression Handler
      BxBlocks bxBlocks;
      bool ZS_enabled = (bool)((block.header().getFlags() >> 1) & 0x01);//getFlags() returns first 8-bits from the amc header
      //std::cout << "ZS_enabled bit: " << ZS_enabled << std::endl; 
      if (ZS_enabled)
	bxBlocks = block.getBxBlocks((unsigned int)6, true);//it returnes 7-32bit bxBlocks originated from the amc13 Block
      else
	bxBlocks = block.getBxBlocks((unsigned int)6, false);//it returnes 6-32bit bxBlocks originated from the amc13 Block
      //std::cout << "BxBlocks collected" << std::endl;
			
      RegionalMuonCandBxCollection *res;
      res = static_cast<BMTFCollections*>(coll)->getBMTFMuons();
      //BxBlocks changed the format of the blocks
      int firstBX = 0, lastBX = 0;
      for (auto bxBlock : bxBlocks) {
      	int bx = bxBlock.header().getBx();
      	if (bx < firstBX)
      	  firstBX = bx;
      }
      lastBX = -firstBX;
      int nBX = lastBX - firstBX + 1;
      //std::cout << "1st bx=" << firstBX << "\tlast bx=" << lastBX << std::endl; 
      res->setBXRange(-2, 2);

      //std::cout << "link = " << (blockId-1)/2 << std::endl;
      //std::cout << "nBX = " << nBX << std::endl;
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

      for (auto bxBlock : bxBlocks) {
	//std::cout << "muons bxBlock to unpack:" << std::endl;
	//testTools::printBlock(bxBlock);
	int ibx = bxBlock.header().getBx();
	//std::cout << "ibx=" << ibx << std::endl;

	for (auto iw = 0; iw < 6; iw+=2) {
	  uint32_t raw_first = bxBlock.payload()[iw];//payload[ip+(ibx+lastBX)*6];
	  uint32_t raw_secnd = bxBlock.payload()[iw+1];//payload[ip+(ibx+lastBX)*6];
	  if ( raw_first == 0 )
	    {
	      //std::cout << "dropped words: " << iw << ", " << iw+1 << std::endl;
	      LogDebug("L1T") << "Raw data is zero";
	      continue;
	    }
					
	  RegionalMuonCand muCand;
	  RegionalMuonRawDigiTranslator::fillRegionalMuonCand(muCand, raw_first, raw_secnd, processor, tftype::bmtf);
	  muCand.setLink(48 + processor);	//the link corresponds to the uGMT input

	  LogDebug("L1T") << "Pt = " << muCand.hwPt() << " eta: " << muCand.hwEta() << " phi: " << muCand.hwPhi();

	  //std::cout << "muon qual = " << muCand.hwQual() << std::endl;
	  if ( muCand.hwQual() != 0 ) {
	    res->push_back(ibx, muCand);
	    //std::cout << "pushed in " << ibx << std::endl;
	  }

	}//for iw
      }//for ibx

      return true;
    }//unpack
  }//ns stage2
}//ns lt1
			
DEFINE_L1T_UNPACKER(l1t::stage2::BMTF_ZSUnpackerOutput);
