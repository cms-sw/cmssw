#include "BMTFPackerOutput.h"

#include <vector>//Panos debug
#include <bitset>//Panos debug

// Implementation
namespace l1t 
{
   namespace stage2 
   {
      Blocks BMTFPackerOutput::pack(const edm::Event& event, const PackerTokens* toks)
      {	 

	int board_id = (int)board();

	std::cout<<"---->\tOutput.cc here"<<std::endl;                                                                                        
	
	auto muonToken = static_cast<const BMTFTokens*>(toks)->getOutputMuonToken();

        Blocks blocks;
        
	edm::Handle<RegionalMuonCandBxCollection> muons;
        event.getByToken(muonToken, muons);

	std::cout<<"board_id = "<< board_id << "\tOutput" <<std::endl;//debug
	//int itrs_over_bx = 0;//debug

	//for(int ibx = muons->getFirstBX(); ibx <= muons->getLastBX(); ibx++)
	//{
	//  itrs_over_bx++;

	    int itrs_over_muons = 0;//debug
	    for (auto imu = muons->begin(); imu != muons->end(); imu++)
	      {
		itrs_over_muons++;
		std::cout<<"imu->processor() = "<<imu->processor()+1<<std::endl;
		std::cout<<"hwPt = " << imu->hwPt() << std::endl;
		std::cout<<"hwSign = " << imu->hwSign() << std::endl;
		std::cout<<"link = " << imu->link() << std::endl;

		if (imu->processor()+1 == board_id){
		  uint32_t firstWord(0), lastWord(0);
		  RegionalMuonRawDigiTranslator::generatePackedDataWords(*imu, firstWord, lastWord);
		  payloadMap_[123].push_back(firstWord); //imu->link()*2+1
		  payloadMap_[123].push_back(lastWord); //imu->link()*2+1
		}
	      }//imu

	    std::cout << "iterations over muons are: " << itrs_over_muons << std::endl;


	    if (payloadMap_[123].size() < 6) //in case less than 3 muons have been found by the processor
	      {
		unsigned int initialSize = payloadMap_[123].size();

		for(unsigned int j = 0; j < 3-initialSize/2; j++){
		  payloadMap_[123].push_back(0);
		  uint32_t nullMuon_word2 = 0 | ( (65532 & 0xFFFF) << 3 ) | ( (2 & 0x3) << 0 );
		  payloadMap_[123].push_back(nullMuon_word2);
		}
	      }
	    else if (payloadMap_[123].size() < 30 && payloadMap_[123].size() > 6)
	      {
		unsigned int initialSize = payloadMap_[123].size();

		for(unsigned int j = 0; j < 15-initialSize/2; j++){
		  payloadMap_[123].push_back(0);
		  uint32_t nullMuon_word2 = 0 | ( (65532 & 0xFFFF) << 3 ) | ( (2 & 0x3) << 0 );
		  payloadMap_[123].push_back(nullMuon_word2);
		}
	      }

	    
	    //  }//ibx
	    //std::cout << "iterations over bx are: " << itrs_over_bx << "\tOutput" << std::endl;
      
	Block block(123, payloadMap_[123]);

	blocks.push_back(block);
	
		
	  //debug from here
	  std::cout << "block id : " << block.header().getID() << std::endl;

	  std::cout << "payload created : " << std::endl;
	  for (auto &word : block.payload())	// 
	    std::cout << std::bitset<32>(word).to_string() << std::endl;
	  //debug up to here
	


         return blocks;
      }

	}//ns stage2
}//ns l1t
DEFINE_L1T_PACKER(l1t::stage2::BMTFPackerOutput);
