#include "DCCTowerBlock.h"
#include "DCCEventBlock.h"
#include "DCCDataParser.h"
#include "DCCXtalBlock.h"
#include "DCCDataMapper.h"
#include "ECALParserBlockException.h"
#include <cstdio>



DCCTBTowerBlock::DCCTBTowerBlock(
 	DCCTBEventBlock * dccBlock, 
	DCCTBDataParser * parser, 
	uint32_t * buffer, 
	uint32_t numbBytes,
	uint32_t wordsToEnd,
	uint32_t wordEventOffset,
	uint32_t expectedTowerID
)
: DCCTBBlockPrototype(parser,"TOWERHEADER", buffer, numbBytes,wordsToEnd, wordEventOffset ) 
 , dccBlock_(dccBlock), expectedTowerID_(expectedTowerID)
{
	
	//Reset error counters ///////////
	errors_["FE::HEADER"]        = 0;
	errors_["FE::TT/SC ID"]      = 0; 
	errors_["FE::BLOCK LENGTH"]  = 0;
	//////////////////////////////////

	
	
	// Get data fields from the mapper and retrieve data /////////////////////////////////////
	mapperFields_ = parser_->mapper()->towerFields();	
	parseData();
	//////////////////////////////////////////////////////////////////////////////////////////
 }
 
 
 void DCCTBTowerBlock::parseXtalData(){
	
	uint32_t numbBytes = blockSize_;
	uint32_t wordsToEnd =wordsToEndOfEvent_;
	
	// See if we can construct the correct number of XTAL Blocks////////////////////////////////////////////////////////////////////////////////
	uint32_t numbDWInXtalBlock = ( parser_->numbXtalSamples() )/4 + 1;
	uint32_t length            = getDataField("BLOCK LENGTH");
	uint32_t numbOfXtalBlocks  = 0 ;
	
	if( length > 0 ){ numbOfXtalBlocks = (length-1)/numbDWInXtalBlock; }
	uint32_t xtalBlockSize     =  numbDWInXtalBlock*8;
	//uint32_t pIncrease         =  numbDWInXtalBlock*2;
	
	//std::cout<<"\n DEBUG::numbDWInXtal Block "<<dec<<numbDWInXtalBlock<<std::endl;
	//std::cout<<"\n DEBUG::length             "<<length<<std::endl;
	//std::cout<<"\n DEBUG::xtalBlockSize      "<<xtalBlockSize<<std::endl;
	//std::cout<<"\n DEBUG::pIncreade          "<<pIncrease<<std::endl;

	
	
	bool zs = dccBlock_->getDataField("ZS");
	if( !zs && numbOfXtalBlocks != 25 ){
	
	   
		(errors_["FE::BLOCK LENGTH"])++;
		errorString_ += "\n ======================================================================\n"; 		
		errorString_ += std::string(" ") + name_ + std::string(" ZS is not active, error in the Tower Length !") ;
		errorString_ += "\n Tower Length is : " + (parser_->getDecString(numbBytes/8))+std::string(" , while it should be : ");
		std::string myString = parser_->getDecString((uint32_t)(25*numbDWInXtalBlock+1));
		errorString_ += "\n It was only possible to build : " + parser_->getDecString( numbOfXtalBlocks)+ std::string(" XTAL blocks");
		errorString_ += "\n ======================================================================";
		blockError_ = true;
	};
	if( numbOfXtalBlocks > 25 ){
		if (errors_["FE::BLOCK LENGTH"]==0)(errors_["FE::BLOCK LENGTH"])++;
		errorString_ += "\n ======================================================================\n"; 		
		errorString_ += std::string(" ") + name_ + std::string(" Tower Length is larger then expected...!") ;
		errorString_ += "\n Tower Length is : " + parser_->getDecString(numbBytes/8)+std::string(" , while it should be at maximum : ");
		std::string myString = parser_->getDecString((uint32_t)(25*numbDWInXtalBlock+1));
		errorString_ += "\n Action -> data after the xtal 25 is ignored... "; 
		errorString_ += "\n ======================================================================";
		blockError_ = true;
		
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	blockSize_     += length*8;  //??????????????????????
	
	// Get XTAL Data //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	uint32_t stripID, xtalID;
	
	
	for(uint32_t numbXtal=1; numbXtal <= numbOfXtalBlocks && numbXtal <=25 ; numbXtal++){
	
		increment(1);
		
		stripID =( numbXtal-1)/5 + 1;	
		xtalID  = numbXtal - (stripID-1)*5;
		
		
		if(!zs){ 	
			xtalBlocks_.push_back(  new DCCTBXtalBlock( parser_, dataP_, xtalBlockSize, wordsToEnd-wordCounter_,wordCounter_+wordEventOffset_,xtalID, stripID) );
		}else{
			xtalBlocks_.push_back(  new DCCTBXtalBlock( parser_, dataP_, xtalBlockSize, wordsToEnd-wordCounter_,wordCounter_+wordEventOffset_,0,0));
		}
		
		increment(xtalBlockSize/4-1);
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		
	// Check internal data ////////////
	if(parser_->debug()){ dataCheck();};
	///////////////////////////////////
}



DCCTBTowerBlock::~DCCTBTowerBlock(){
	std::vector<DCCTBXtalBlock *>::iterator it;
	for(it=xtalBlocks_.begin();it!=xtalBlocks_.end();it++){ delete (*it);}
	xtalBlocks_.clear();
}



void DCCTBTowerBlock::dataCheck(){
	std::string checkErrors("");	
	
	
	std::pair <bool,std::string> res;
	
	///////////////////////////////////////////////////////////////////////////
	// For TB we don-t check Bx 
	//res = checkDataField("BX", BXMASK & (dccBlock_->getDataField("BX")));
	//if(!res.first){ checkErrors += res.second; (errors_["FE::HEADER"])++; }
	////////////////////////////////////////////////////////////////////////////
	
        // mod to account for ECAL counters starting from 0 in the front end N. Almeida
	res = checkDataField("LV1", L1MASK &  (dccBlock_->getDataField("LV1")  -1)   ); 
	if(!res.first){ checkErrors += res.second; (errors_["FE::HEADER"])++; }
	
	
	if(expectedTowerID_ != 0){ 
		res = checkDataField("TT/SC ID",expectedTowerID_); 
		if(!res.first){ checkErrors += res.second; (errors_["FE::HEADER"])++; } 
	}
	
	if( checkErrors !="" ){
		std::string myTowerId;
		
		errorString_ +="\n ======================================================================\n"; 
		errorString_ += std::string(" ") + name_ + std::string("( ID = ")+parser_->getDecString((uint32_t)(expectedTowerID_))+std::string(" ) errors : ") ;
		errorString_ += checkErrors ;
		errorString_ += "\n ======================================================================";
		blockError_ = true;	
	}
} 


std::vector< DCCTBXtalBlock * > DCCTBTowerBlock::xtalBlocksById(uint32_t stripId, uint32_t xtalId){
	std::vector<DCCTBXtalBlock *> myVector;	
	std::vector<DCCTBXtalBlock *>::iterator it;
	
	for( it = xtalBlocks_.begin(); it!= xtalBlocks_.end(); it++ ){
		try{
			
		  std::pair<bool,std::string> stripIdCheck   = (*it)->checkDataField("STRIP ID",stripId);
		  std::pair<bool,std::string> xtalIdCheck    = (*it)->checkDataField("XTAL ID",xtalId);
			
			if(xtalIdCheck.first && stripIdCheck.first ){ myVector.push_back( (*it) ); }
			
		}catch (ECALTBParserBlockException &e){/*ignore*/ }
	}
	
	return myVector;
}

int DCCTBTowerBlock::towerID() {
  int result=-1;

  for(std::set<DCCTBDataField *,DCCTBDataFieldComparator>::iterator it = mapperFields_->begin(); it!= mapperFields_->end(); it++){
    if ( (*it)->name() == "TT/SC ID" ) 
      result=getDataField( (*it)->name() )  ;
    
  }
  
  return result;

}
