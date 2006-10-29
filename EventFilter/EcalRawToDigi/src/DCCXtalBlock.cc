#include "DCCXtalBlock.h"
#include "DCCDataParser.h"
#include "DCCDataMapper.h"


DCCXtalBlock::DCCXtalBlock(
	DCCDataParser * parser, 
	ulong * buffer, 
	ulong numbBytes,  
	ulong wordsToEnd,
	ulong wordEventOffset,
	ulong expectedXtalID,
	ulong expectedStripID
) : DCCBlockPrototype(parser,"XTAL", buffer, numbBytes, wordsToEnd, wordEventOffset),
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



void DCCXtalBlock::dataCheck(){
	
	string checkErrors("");
		
	pair <bool,string> res;
	
	
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
		errorString_ += string(" ") + name_ + string(" data fields checks errors : ") ;
		errorString_ += checkErrors ;
		errorString_ += "\n ======================================================================";
		blockError_ = true;	
	}
	
}


void  DCCXtalBlock::increment(ulong numb){
	if(!parser_->debug()){ DCCBlockPrototype::increment(numb); }
	else {
		for(ulong counter=0; counter<numb; counter++, dataP_++,wordCounter_++){
			ulong blockID = (*dataP_)>>BPOSITION_BLOCKID;
			if( blockID != BLOCKID ){
				(errors_["XTAL::BLOCKID"])++;
				//errorString_ += string("\n") + parser_->index(nunb)+(" blockId has value ") + parser_->getDecString(blockID);
				//errorString  += string(", while ")+parser_->getDecString(BLOCKID)+string(" is expected");
			}
		}
	}
}

int DCCXtalBlock::xtalID() {

  int result=-1;

  for(  set<DCCDataField *,DCCDataFieldComparator>::iterator 
           it = mapperFields_->begin(); it!= mapperFields_->end(); it++){
  
    if ( (*it)->name() == "XTAL ID" ) 
      result=getDataField( (*it)->name() )  ;
    
  }


 
  return result; 

}

int DCCXtalBlock::stripID() {
  int result=-1;

  for(set<DCCDataField *,DCCDataFieldComparator>::iterator it = mapperFields_->begin(); it!= mapperFields_->end(); it++){
    if ( (*it)->name() == "STRIP ID" ) 
      result=getDataField( (*it)->name() )  ;
    
  }
  
  return result;

}



vector<int> DCCXtalBlock::xtalDataSamples() {
  vector<int> data;


  for(unsigned int i=1;i <= parser_->numbXtalSamples();i++){
    string name = string("ADC#") + parser_->getDecString(i);
    
    data.push_back ( getDataField( name )  );
     
  }

  return data;
}
