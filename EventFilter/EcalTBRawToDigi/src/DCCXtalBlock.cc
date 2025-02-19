#include "DCCXtalBlock.h"
#include "DCCDataParser.h"
#include "DCCDataMapper.h"


DCCTBXtalBlock::DCCTBXtalBlock(
	DCCTBDataParser * parser, 
	uint32_t * buffer, 
	uint32_t numbBytes,  
	uint32_t wordsToEnd,
	uint32_t wordEventOffset,
	uint32_t expectedXtalID,
	uint32_t expectedStripID
) : DCCTBBlockPrototype(parser,"XTAL", buffer, numbBytes, wordsToEnd, wordEventOffset),
expectedXtalID_(expectedXtalID), expectedStripID_(expectedStripID){
	
	
	//Reset error counters ////
	errors_["XTAL::HEADER"]  = 0;
	errors_["XTAL::BLOCKID"] = 0; 
	///////////////////////////
	
	// Get data fields from the mapper and retrieve data /////////////////////////////////////
	mapperFields_ = parser_->mapper()->xtalFields();	
	parseData();
	//////////////////////////////////////////////////////////////////////////////////////////
	
	// check internal data ////////////
	if(parser_->debug()){ dataCheck();}
	///////////////////////////////////

}



void DCCTBXtalBlock::dataCheck(){
	
	std::string checkErrors("");
		
	std::pair <bool,std::string> res;
	
	
	if(expectedXtalID_ !=0){ 
		res = checkDataField("XTAL ID",expectedXtalID_); 
		if(!res.first){ checkErrors += res.second; (errors_["XTAL::HEADER"])++; } 
	}
	if(expectedStripID_!=0){ 
		res = checkDataField("STRIP ID",expectedStripID_);
		if(!res.first){ checkErrors += res.second; (errors_["XTAL::HEADER"])++; } 
	}
	if(checkErrors!=""){
		errorString_ +="\n ======================================================================\n"; 		
		errorString_ += std::string(" ") + name_ + std::string(" data fields checks errors : ") ;
		errorString_ += checkErrors ;
		errorString_ += "\n ======================================================================";
		blockError_ = true;	
	}
	
}


void  DCCTBXtalBlock::increment(uint32_t numb){
	if(!parser_->debug()){ DCCTBBlockPrototype::increment(numb); }
	else {
		for(uint32_t counter=0; counter<numb; counter++, dataP_++,wordCounter_++){
			uint32_t blockID = (*dataP_)>>BPOSITION_BLOCKID;
			if( blockID != BLOCKID ){
				(errors_["XTAL::BLOCKID"])++;
				//errorString_ += std::string("\n") + parser_->index(nunb)+(" blockId has value ") + parser_->getDecString(blockID);
				//errorString  += std::string(", while ")+parser_->getDecString(BLOCKID)+std::string(" is expected");
			}
		}
	}
}

int DCCTBXtalBlock::xtalID() {

  int result=-1;

  for(  std::set<DCCTBDataField *,DCCTBDataFieldComparator>::iterator 
           it = mapperFields_->begin(); it!= mapperFields_->end(); it++){
  
    if ( (*it)->name() == "XTAL ID" ) 
      result=getDataField( (*it)->name() )  ;
    
  }


 
  return result; 

}

int DCCTBXtalBlock::stripID() {
  int result=-1;

  for(std::set<DCCTBDataField *,DCCTBDataFieldComparator>::iterator it = mapperFields_->begin(); it!= mapperFields_->end(); it++){
    if ( (*it)->name() == "STRIP ID" ) 
      result=getDataField( (*it)->name() )  ;
    
  }
  
  return result;

}



std::vector<int> DCCTBXtalBlock::xtalDataSamples() {
  std::vector<int> data;


  for(unsigned int i=1;i <= parser_->numbXtalSamples();i++){
    std::string name = std::string("ADC#") + parser_->getDecString(i);
    
    data.push_back ( getDataField( name )  );
     
  }

  return data;
}
