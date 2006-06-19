#include "EventFilter/EcalRawToDigi/src/DCCSRPBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataParser.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataMapper.h"
#include "EventFilter/EcalRawToDigi/src/DCCEventBlock.h"

DCCSRPBlock::DCCSRPBlock(
	DCCEventBlock * dccBlock,
	DCCDataParser * parser, 
	ulong * buffer, 
	ulong numbBytes,
	ulong wordsToEnd,
	ulong wordEventOffset
) : DCCBlockPrototype(parser,"SRP", buffer, numbBytes,wordsToEnd,wordEventOffset), dccBlock_(dccBlock){
	
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



void DCCSRPBlock::dataCheck(){ 
	
	string checkErrors("");

	pair <bool,string> res;
	
	res = checkDataField("BX", BXMASK & (dccBlock_->getDataField("BX")));
	if(!res.first){ checkErrors += res.second; (errors_["SRP::HEADER"])++; }
	res = checkDataField("LV1", L1MASK & (dccBlock_->getDataField("LV1"))); 
	if(!res.first){ checkErrors += res.second; (errors_["SRP::HEADER"])++; }
	
	 
	res = checkDataField("SRP ID",parser_->srpId()); 
	if(!res.first){ checkErrors += res.second; (errors_["SRP::HEADER"])++; } 
	
	
	if(checkErrors!=""){
		errorString_ +="\n ======================================================================\n"; 		
		errorString_ += string(" ") + name_ + string(" data fields checks errors : ") ;
		errorString_ += checkErrors ;
		errorString_ += "\n ======================================================================";
		blockError_ = true;	
	}
	
	
}


void  DCCSRPBlock::increment(ulong numb){
	if(!parser_->debug()){ DCCBlockPrototype::increment(numb); }
	else {
		for(ulong counter=0; counter<numb; counter++, dataP_++,wordCounter_++){
			ulong blockID = (*dataP_)>>BPOSITION_BLOCKID;
			if( blockID != BLOCKID ){
				(errors_["SRP::BLOCKID"])++;
				//errorString_ += string("\n") + parser_->index(nunb)+(" blockId has value ") + parser_->getDecString(blockID);
				//errorString  += string(", while ")+parser_->getDecString(BLOCKID)+string(" is expected");
			}
		}
	}
	
	
}


