#include "EventFilter/EcalRawToDigi/src/DCCTowerBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataParser.h"
#include "EventFilter/EcalRawToDigi/src/DCCXtalBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataMapper.h"
#include "ECALParserBlockException.h"
#include <stdio.h>



DCCTowerBlock::DCCTowerBlock(
 	DCCEventBlock * dccBlock, 
	DCCDataParser * parser, 
	ulong * buffer, 
	ulong numbBytes,
	ulong wordsToEnd,
	ulong wordEventOffset,
	ulong expectedTowerID
)
: DCCBlockPrototype(parser,"TOWERHEADER", buffer, numbBytes,wordsToEnd, wordEventOffset ) 
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
 
 
 void DCCTowerBlock::parseXtalData(){
	
	ulong numbBytes = blockSize_;
	ulong wordsToEnd =wordsToEndOfEvent_;
	
	// See if we can construct the correct number of XTAL Blocks////////////////////////////////////////////////////////////////////////////////
	ulong numbDWInXtalBlock = ( parser_->numbXtalSamples() )/4 + 1;
	ulong length            = getDataField("BLOCK LENGTH");
	ulong numbOfXtalBlocks  = 0 ;
	
	if( length > 0 ){ numbOfXtalBlocks = (length-1)/numbDWInXtalBlock; }
	ulong xtalBlockSize     =  numbDWInXtalBlock*8;
	//ulong pIncrease         =  numbDWInXtalBlock*2;
	
	//cout<<"\n DEBUG::numbDWInXtal Block "<<dec<<numbDWInXtalBlock<<endl;
	//cout<<"\n DEBUG::length             "<<length<<endl;
	//cout<<"\n DEBUG::xtalBlockSize      "<<xtalBlockSize<<endl;
	//cout<<"\n DEBUG::pIncreade          "<<pIncrease<<endl;

	
	
	bool zs = dccBlock_->getDataField("ZS");
	if( !zs && numbOfXtalBlocks != 25 ){
	
	   
		(errors_["FE::BLOCK LENGTH"])++;
		errorString_ += "\n ======================================================================\n"; 		
		errorString_ += string(" ") + name_ + string(" ZS is not active, error in the Tower Length !") ;
		errorString_ += "\n Tower Length is : " + (parser_->getDecString(numbBytes/8))+string(" , while it should be : ");
		string myString = parser_->getDecString((ulong)(25*numbDWInXtalBlock+1));
		errorString_ += "\n It was only possible to build : " + parser_->getDecString( numbOfXtalBlocks)+ string(" XTAL blocks");
		errorString_ += "\n ======================================================================";
		blockError_ = true;
	};
	if( numbOfXtalBlocks > 25 ){
		if (errors_["FE::BLOCK LENGTH"]==0)(errors_["FE::BLOCK LENGTH"])++;
		errorString_ += "\n ======================================================================\n"; 		
		errorString_ += string(" ") + name_ + string(" Tower Length is larger then expected...!") ;
		errorString_ += "\n Tower Length is : " + parser_->getDecString(numbBytes/8)+string(" , while it should be at maximum : ");
		string myString = parser_->getDecString((ulong)(25*numbDWInXtalBlock+1));
		errorString_ += "\n Action -> data after the xtal 25 is ignored... "; 
		errorString_ += "\n ======================================================================";
		blockError_ = true;
		
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	blockSize_     += length*8;  //??????????????????????
	
	// Get XTAL Data //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	ulong stripID, xtalID;
	
	
	for(ulong numbXtal=1; numbXtal <= numbOfXtalBlocks && numbXtal <=25 ; numbXtal++){
	
		increment(1);
		
		stripID =( numbXtal-1)/5 + 1;	
		xtalID  = numbXtal - (stripID-1)*5;
		
		
		if(!zs){ 	
			xtalBlocks_.push_back(  new DCCXtalBlock( parser_, dataP_, xtalBlockSize, wordsToEnd-wordCounter_,wordCounter_+wordEventOffset_,xtalID, stripID) );
		}else{
			xtalBlocks_.push_back(  new DCCXtalBlock( parser_, dataP_, xtalBlockSize, wordsToEnd-wordCounter_,wordCounter_+wordEventOffset_,0,0));
		}
		
		increment(xtalBlockSize/4-1);
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		
	// Check internal data ////////////
	if(parser_->debug()){ dataCheck();};
	///////////////////////////////////
}



DCCTowerBlock::~DCCTowerBlock(){
	vector<DCCXtalBlock *>::iterator it;
	for(it=xtalBlocks_.begin();it!=xtalBlocks_.end();it++){ delete (*it);}
	xtalBlocks_.clear();
}



void DCCTowerBlock::dataCheck(){
	string checkErrors("");	
	
	
	pair <bool,string> res;
	
	///////////////////////////////////////////////////////////////////////////
	// For TB we don-t check Bx 
	//res = checkDataField("BX", BXMASK & (dccBlock_->getDataField("BX")));
	//if(!res.first){ checkErrors += res.second; (errors_["FE::HEADER"])++; }
	////////////////////////////////////////////////////////////////////////////
	
	res = checkDataField("LV1", L1MASK & (dccBlock_->getDataField("LV1"))); 
	if(!res.first){ checkErrors += res.second; (errors_["FE::HEADER"])++; }
	
	
	if(expectedTowerID_ != 0){ 
		res = checkDataField("TT/SC ID",expectedTowerID_); 
		if(!res.first){ checkErrors += res.second; (errors_["FE::HEADER"])++; } 
	}
	
	if( checkErrors !="" ){
		errorString_ +="\n ======================================================================\n"; 
		errorString_ += string(" ") + name_ + string(" data fields checks errors : ") ;
		errorString_ += checkErrors ;
		errorString_ += "\n ======================================================================";
		blockError_ = true;	
	}
} 


vector< DCCXtalBlock * > DCCTowerBlock::xtalBlocksById(ulong stripId, ulong xtalId){
	vector<DCCXtalBlock *> myVector;	
	vector<DCCXtalBlock *>::iterator it;
	
	for( it = xtalBlocks_.begin(); it!= xtalBlocks_.end(); it++ ){
		try{
			
			pair<bool,string> stripIdCheck   = (*it)->checkDataField("STRIP ID",stripId);
			pair<bool,string> xtalIdCheck    = (*it)->checkDataField("XTAL ID",xtalId);
			
			if(xtalIdCheck.first && stripIdCheck.first ){ myVector.push_back( (*it) ); }
			
		}catch (ECALParserBlockException &e){/*ignore*/ }
	}
	
	return myVector;
}



int DCCTowerBlock::towerID() {
  int result=-1;

  for(set<DCCDataField *,DCCDataFieldComparator>::iterator it = mapperFields_->begin(); it!= mapperFields_->end(); it++){
    if ( (*it)->name() == "TT/SC ID" ) 
      result=getDataField( (*it)->name() )  ;
    
  }
  
  return result;

}
