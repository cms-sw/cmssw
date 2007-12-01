/*--------------------------------------------------------------*/
/* DCC TCC BLOCK CLASS                                          */
/*                                                              */
/* Author : N.Almeida (LIP)  Date   : 30/05/2005                */
/*--------------------------------------------------------------*/

#include "DCCTCCBlock.h"

/*-------------------------------------------------*/
/* DCCTCCBlock::DCCTCCBlock                        */
/* class constructor                               */
/*-------------------------------------------------*/
DCCTCCBlock::DCCTCCBlock(
	DCCEventBlock * dccBlock,
	DCCDataParser * parser, 
	ulong * buffer, 
	ulong numbBytes,  
	ulong wordsToEnd,
	ulong wordEventOffset,
	ulong expectedId) : 
  DCCBlockPrototype(parser,"TCC", buffer, numbBytes, wordsToEnd, wordEventOffset),dccBlock_(dccBlock), expectedId_(expectedId){

  //Reset error counters
  errors_["TCC::HEADER"]  = 0;
  errors_["TCC::BLOCKID"] = 0;
	
  //Get data fields from the mapper and retrieve data 
  if(      parser_->numbTTs() == 68){ mapperFields_ = parser_->mapper()->tcc68Fields();}
  else if( parser_->numbTTs() == 32){ mapperFields_ = parser_->mapper()->tcc32Fields();}	
  else if( parser_->numbTTs() == 16){ mapperFields_ = parser_->mapper()->tcc16Fields();}
  
  parseData();
  
  // check internal data 
  if(parser_->debug())
    dataCheck();
}
 
/*---------------------------------------------------*/
/* DCCTCCBlock::dataCheck                            */
/* check data with data fields                       */
/*---------------------------------------------------*/
void DCCTCCBlock::dataCheck(){
  std::pair <bool,std::string> res;            //check result
  std::string checkErrors("");            //error string

  //check BX(LOCAL) field (1st word bit 16)
  res = checkDataField("BX", BXMASK & (dccBlock_->getDataField("BX")));
  if(!res.first){ 
    checkErrors += res.second; 
    (errors_["TCC::HEADER"])++; 
  }
  
  //check LV1(LOCAL) field (1st word bit 32)
  res = checkDataField("LV1", L1MASK & (dccBlock_->getDataField("LV1"))); 
  if(!res.first){ 
    checkErrors += res.second; 
    (errors_["TCC::HEADER"])++; 
  }
  
  //check TCC ID field (1st word bit 0)
  res = checkDataField("TCC ID",expectedId_); 
  if(!res.first){ 
    checkErrors += res.second; 
    (errors_["TCC::HEADER"])++; 
  } 
  
  
  if(checkErrors!=""){
  	 blockError_=true;
    errorString_ +="\n ======================================================================\n"; 
	 errorString_ += std::string(" ") + name_ + std::string("( ID = ")+parser_->getDecString((ulong)(expectedId_))+std::string(" ) errors : ") ;
	 errorString_ += checkErrors ;
	 errorString_ += "\n ======================================================================";
  }
  
  
}


/*--------------------------------------------------*/
/* DCCTCCBlock::increment                           */
/* increment a TCC block                            */
/*--------------------------------------------------*/

void  DCCTCCBlock::increment(ulong numb){
  //if no debug is required increments the number of blocks
  //otherwise checks if block id is really B'011'=3
  if(!parser_->debug()){ 
    DCCBlockPrototype::increment(numb); 
  }
  else {
    for(ulong counter=0; counter<numb; counter++, dataP_++, wordCounter_++){
      ulong blockID = (*dataP_) >> BPOSITION_BLOCKID;
      if( blockID != BLOCKID ){
	(errors_["TCC::BLOCKID"])++;
	//errorString_ += std::string("\n") + parser_->index(nunb)+(" blockId has value ") + parser_->getDecString(blockID);
	//errorString  += std::string(", while ")+parser_->getDecString(BLOCKID)+std::string(" is expected");
      }
    }
  }
  
}



std::vector< std::pair<int,bool> > DCCTCCBlock::triggerSamples() {
  std::vector< std::pair<int,bool> > data;

  for(unsigned int i=1;i <= parser_->numbTTs();i++){
    std::string name = std::string("TPG#") + parser_->getDecString(i);
    int tpgValue = getDataField( name ) ;
                 std::pair<int,bool> tpg( tpgValue&ETMASK, bool(tpgValue>>BPOSITION_FGVB));
    data.push_back (tpg);
     
  }

  return data;
}




std::vector<int> DCCTCCBlock::triggerFlags() {
  std::vector<int> data;

  for(unsigned int i=1; i<= parser_->numbTTs();i++){
    std::string name = std::string("TTF#") + parser_->getDecString(i);
    
    data.push_back ( getDataField( name )  );
     
  }

  return data;
}




