#include "DCCTrailerBlock.h"
#include "DCCDataParser.h"
#include "DCCDataMapper.h"
DCCTBTrailerBlock::DCCTBTrailerBlock(
	DCCTBDataParser * parser, 
	uint32_t * buffer, 
	uint32_t numbBytes,  
	uint32_t wToEnd,
	uint32_t wordEventOffset,
	uint32_t expectedLength,
	uint32_t expectedCRC
) : DCCTBBlockPrototype(parser,"DCCTRAILER", buffer, numbBytes,wToEnd, wordEventOffset),
expectedLength_(expectedLength){
	
	errors_["TRAILER::EVENT LENGTH"] = 0 ;
	errors_["TRAILER::EOE"]    = 0 ; 
	errors_["TRAILER::CRC"]    = 0 ;
	errors_["TRAILER::T"]      = 0 ;
	
	// Get data fields from the mapper and retrieve data ///////////////////////////////////////////
	mapperFields_ = parser_->mapper()->trailerFields();
	parseData();
	////////////////////////////////////////////////////////////////////////////////////////////////

	// check internal data ////
	dataCheck();
	///////////////////////////

}


void DCCTBTrailerBlock::dataCheck(){
	
	std::string checkErrors("");
	
	std::pair<bool,std::string> res;
	
	res = checkDataField("EVENT LENGTH",expectedLength_);
	if(!res.first){ checkErrors += res.second; (errors_["TRAILER::EVENT LENGTH"])++; }
	
	res = checkDataField("EOE",EOE);
	if(!res.first){ checkErrors += res.second; (errors_["TRAILER::EOE"])++; }
	
	res = checkDataField("T",0);
	if(!res.first){ checkErrors += res.second; (errors_["TRAILER::T"])++; }
	
	//checkErrors += checkDataField("CRC",expectedCRC_);
	
	if(checkErrors!=""){
		errorString_ +="\n ======================================================================\n"; 		
		errorString_ += std::string(" ") + name_ + std::string(" data fields checks errors : ") ;
		errorString_ += checkErrors ;
		errorString_ += "\n ======================================================================";
		blockError_ = true;	
	}
}

