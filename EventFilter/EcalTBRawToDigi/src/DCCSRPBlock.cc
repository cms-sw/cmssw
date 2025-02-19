#include "DCCSRPBlock.h"
#include "DCCDataParser.h"
#include "DCCDataMapper.h"
#include "DCCEventBlock.h"

DCCTBSRPBlock::DCCTBSRPBlock(
	DCCTBEventBlock * dccBlock,
	DCCTBDataParser * parser, 
	uint32_t * buffer, 
	uint32_t numbBytes,
	uint32_t wordsToEnd,
	uint32_t wordEventOffset
) : DCCTBBlockPrototype(parser,"SRP", buffer, numbBytes,wordsToEnd,wordEventOffset), dccBlock_(dccBlock){
	
	//Reset error counters ///////
	errors_["SRP::HEADER"]  = 0;
	errors_["SRP::BLOCKID"] = 0;
	//////////////////////////////
	
	// Get data fields from the mapper and retrieve data /////////////////////////////////////
	     if( parser_->numbSRF() == 68){ mapperFields_ = parser_->mapper()->srp68Fields();}
	else if( parser_->numbSRF() == 32){ mapperFields_ = parser_->mapper()->srp32Fields();}	
	else if( parser_->numbSRF() == 16){ mapperFields_ = parser_->mapper()->srp16Fields();}
	parseData();
	//////////////////////////////////////////////////////////////////////////////////////////
	
	// check internal data ////////////
	if(parser_->debug()){ dataCheck();}
	///////////////////////////////////

}



void DCCTBSRPBlock::dataCheck(){ 
	
	std::string checkErrors("");

	std::pair <bool,std::string> res;
	
	res = checkDataField("BX", BXMASK & (dccBlock_->getDataField("BX")));
	if(!res.first){ checkErrors += res.second; (errors_["SRP::HEADER"])++; }
	res = checkDataField("LV1", L1MASK & (dccBlock_->getDataField("LV1"))); 
	if(!res.first){ checkErrors += res.second; (errors_["SRP::HEADER"])++; }
	
	 
	res = checkDataField("SRP ID",parser_->srpId()); 
	if(!res.first){ checkErrors += res.second; (errors_["SRP::HEADER"])++; } 
	
	
	if(checkErrors!=""){
		errorString_ +="\n ======================================================================\n"; 		
		errorString_ += std::string(" ") + name_ + std::string(" data fields checks errors : ") ;
		errorString_ += checkErrors ;
		errorString_ += "\n ======================================================================";
		blockError_ = true;	
	}
	
	
}


void  DCCTBSRPBlock::increment(uint32_t numb){
	if(!parser_->debug()){ DCCTBBlockPrototype::increment(numb); }
	else {
		for(uint32_t counter=0; counter<numb; counter++, dataP_++,wordCounter_++){
			uint32_t blockID = (*dataP_)>>BPOSITION_BLOCKID;
			if( blockID != BLOCKID ){
				(errors_["SRP::BLOCKID"])++;
				//errorString_ += std::string("\n") + parser_->index(nunb)+(" blockId has value ") + parser_->getDecString(blockID);
				//errorString  += std::string(", while ")+parser_->getDecString(BLOCKID)+std::string(" is expected");
			}
		}
	}
	
	
}


