#include "EventFilter/EcalRawToDigi/src/DCCTrailerBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataParser.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataMapper.h"
DCCTrailerBlock::DCCTrailerBlock(
	DCCDataParser * parser, 
	ulong * buffer, 
	ulong numbBytes,  
	ulong wToEnd,
	ulong wordEventOffset,
	ulong expectedLength,
	ulong expectedCRC
) : DCCBlockPrototype(parser,"DCCTRAILER", buffer, numbBytes,wToEnd, wordEventOffset),
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


void DCCTrailerBlock::dataCheck(){
	
	string checkErrors("");
	
	pair<bool,string> res;
	
	res = checkDataField("EVENT LENGTH",expectedLength_);
	if(!res.first){ checkErrors += res.second; (errors_["TRAILER::EVENT LENGTH"])++; }
	
	res = checkDataField("EOE",EOE);
	if(!res.first){ checkErrors += res.second; (errors_["TRAILER::EOE"])++; }
	
	res = checkDataField("T",0);
	if(!res.first){ checkErrors += res.second; (errors_["TRAILER::T"])++; }
	
	//checkErrors += checkDataField("CRC",expectedCRC_);
	
	if(checkErrors!=""){
		errorString_ +="\n ======================================================================\n"; 		
		errorString_ += string(" ") + name_ + string(" data fields checks errors : ") ;
		errorString_ += checkErrors ;
		errorString_ += "\n ======================================================================";
		blockError_ = true;	
	}
}

