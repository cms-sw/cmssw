#include "DCCTCCBlock.h"
#include "DCCDataParser.h"
#include "DCCDataMapper.h"
#include "DCCEventBlock.h"

DCCTCCBlock::DCCTCCBlock(
	DCCEventBlock * dccBlock,
	DCCDataParser * parser, 
	ulong * buffer, 
	ulong numbBytes,  
	ulong wordsToEnd,
	ulong wordEventOffset,
	ulong expectedId
) : DCCBlockPrototype(parser,"TCC", buffer, numbBytes, wordsToEnd, wordEventOffset), 
dccBlock_(dccBlock), expectedId_(expectedId){

	//Reset error counters ////
	errors_["TCC::HEADER"]  = 0;
	errors_["TCC::BLOCKID"] = 0;
	///////////////////////////

	
	// Get data fields from the mapper and retrieve data /////////////////////////////////////
	     if( parser_->numbTTs() == 68){ mapperFields_ = parser_->mapper()->tcc68Fields();}
	else if( parser_->numbTTs() == 32){ mapperFields_ = parser_->mapper()->tcc32Fields();}	
	else if( parser_->numbTTs() == 16){ mapperFields_ = parser_->mapper()->tcc16Fields();}
	parseData();
	//////////////////////////////////////////////////////////////////////////////////////////


	// check internal data ////////////
	if(parser_->debug()){ dataCheck();}
	///////////////////////////////////

}
 

void DCCTCCBlock::dataCheck(){
	
	string checkErrors("");

	pair <bool,string> res;
	
	res = checkDataField("BX", BXMASK & (dccBlock_->getDataField("BX")));
	if(!res.first){ checkErrors += res.second; (errors_["TCC::HEADER"])++; }
	res = checkDataField("LV1", L1MASK & (dccBlock_->getDataField("LV1"))); 
	if(!res.first){ checkErrors += res.second; (errors_["TCC::HEADER"])++; }

	res = checkDataField("TCC ID",expectedId_); 
	if(!res.first){ checkErrors += res.second; (errors_["TCC::HEADER"])++; } 
	

	errorString_ += checkErrors;
	if(checkErrors!=""){ blockError_=true; }
	
}


void  DCCTCCBlock::increment(ulong numb){
	if(!parser_->debug()){ DCCBlockPrototype::increment(numb); }
	else {
			for(ulong counter=0; counter<numb; counter++, dataP_++,wordCounter_++){
				ulong blockID = (*dataP_)>>BPOSITION_BLOCKID;
				if( blockID != BLOCKID ){
					(errors_["TCC::BLOCKID"])++;
					//errorString_ += string("\n") + parser_->index(nunb)+(" blockId has value ") + parser_->getDecString(blockID);
					//errorString  += string(", while ")+parser_->getDecString(BLOCKID)+string(" is expected");
				}
			}
	}

}

