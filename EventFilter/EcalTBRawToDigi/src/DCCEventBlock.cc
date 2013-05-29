#include "DCCEventBlock.h"
#include "DCCDataParser.h"
#include "DCCDataMapper.h"
#include "DCCTowerBlock.h"
#include "ECALParserException.h"
#include "ECALParserBlockException.h"
#include "DCCSRPBlock.h"
#include "DCCTCCBlock.h"
#include "DCCXtalBlock.h"
#include "DCCTrailerBlock.h"

#include <iomanip>
#include <sstream>

DCCTBEventBlock::DCCTBEventBlock(
	DCCTBDataParser * parser, 
	uint32_t * buffer, 
	uint32_t numbBytes, 
	uint32_t wordsToEnd, 
	uint32_t wordBufferOffset , 
	uint32_t wordEventOffset 
) : 
DCCTBBlockPrototype(parser,"DCCHEADER", buffer, numbBytes,wordsToEnd)
,dccTrailerBlock_(0),srpBlock_(0),wordBufferOffset_(wordBufferOffset) {
	
	
	//Reset error counters ////
	errors_["DCC::HEADER"] = 0;
	errors_["DCC::EVENT LENGTH"] = 0;
	///////////////////////////
	
	uint32_t wToEnd(0);
	
	try{ 
	
		// Get data fields from the mapper and retrieve data ///	
		if(numbBytes == DCCTBDataParser::EMPTYEVENTSIZE ){
			mapperFields_ = parser_->mapper()->emptyEventFields();
			emptyEvent = true;
		}else{
			mapperFields_ = parser_->mapper()->dccFields();	
			emptyEvent = false;
		}
		
		try{ parseData(); }
		catch (ECALTBParserBlockException &e){/*ignore*/}
		///////////////////////////////////////////////////////
		

	
		// Check internal data //////////////
		if(parser_->debug()){ dataCheck(); }
		/////////////////////////////////////
	
	
		// Check if empty event was produced /////////////////////////////////////////////////////////////////

		if( !emptyEvent && getDataField("DCC ERRORS")!= DCCERROR_EMPTYEVENT ){
			 		
			// Build the SRP block ////////////////////////////////////////////////////////////////////////////////////
		
			bool srp(false);
			uint32_t sr_ch = getDataField("SR_CHSTATUS");
			if( sr_ch!=CH_TIMEOUT  && sr_ch != CH_DISABLED ){ 			
				
				//Go to the begining of the block
				increment(1," (while trying to create a SR Block !)");
				wToEnd = numbBytes/4-wordCounter_-1;	
				
				// Build SRP Block //////////////////////////////////////////////////////////////////////
				srpBlock_ = new DCCTBSRPBlock( this, parser_, dataP_, parser_->srpBlockSize(), wToEnd,wordCounter_);
				//////////////////////////////////////////////////////////////////////////////////////////
		
				increment((parser_->srpBlockSize())/4-1);
				if(getDataField("SR")){ srp=true; }
			}	
			
			////////////////////////////////////////////////////////////////////////////////////////////////////////////		


			// Build TCC blocks ////////////////////////////////////////////////////////////////////////////////////////
			for(uint32_t i=1; i<=4;i++){
			  uint32_t tcc_ch=0;  uint32_t  tccId=0;
				if( i == 1){ tccId = parser_->tcc1Id();}
				if( i == 2){ tccId = parser_->tcc2Id();}
				if( i == 3){ tccId = parser_->tcc3Id();}	
				if( i == 4){ tccId = parser_->tcc4Id();}
				
				std::string tcc = std::string("TCC_CHSTATUS#") + parser_->getDecString(i);	
				tcc_ch = getDataField(tcc);
				
				if( tcc_ch != CH_TIMEOUT && tcc_ch != CH_DISABLED){	 
					
					//std::cout<<"\n debug:Building TCC Block, channel enabled without errors"<<std::endl;
					
					// Go to the begining of the block
					increment(1," (while trying to create a"+tcc+" Block !)");
					
					wToEnd = numbBytes/4-wordCounter_-1;	
					//wToEnd or wordsToEnd ????????????????????????????????????????
					
				
					
					// Build TCC Block /////////////////////////////////////////////////////////////////////////////////
					tccBlocks_.push_back(  new DCCTBTCCBlock( this, parser_, dataP_,parser_->tccBlockSize(), wToEnd,wordCounter_, tccId));
					//////////////////////////////////////////////////////////////////////////////////////////////////////	
					
					increment((parser_->tccBlockSize())/4-1);
				}
			}
			////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
			
			
			// Build channel data //////////////////////////////////////////////////////////////////////////////////////////////////////	
			// See number of channels that we need according to the trigger type //
			// TODO : WHEN IN LOCAL MODE WE SHOULD CHECK RUN TYPE
			uint32_t triggerType = getDataField("TRIGGER TYPE");			
			uint32_t numbChannels;
			if( triggerType == PHYSICTRIGGER )          { numbChannels = 68; }
			else if (triggerType == CALIBRATIONTRIGGER ){ numbChannels = 70; }
			// TODO :: implement other triggers
			else{
			  std::string error = std::string("\n DCC::HEADER TRIGGER TYPE = ")+parser_->getDecString(triggerType)+std::string(" is not a valid type !");
				ECALTBParserBlockException a(error);
				throw a;
			}
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
			
			
//			uint32_t chStatus;
			uint32_t srFlag;
			bool suppress(false);
			
			for( uint32_t i=1; i<=numbChannels; i++){
				
			  std::string chStatusId = std::string("FE_CHSTATUS#") + parser_->getDecString(i);
				uint32_t chStatus = getDataField(chStatusId);
				
				// If srp is on, we need to check if channel was suppressed ////////////////////
				if(srp){ 
					srFlag   = srpBlock_->getDataField( std::string("SR#") + parser_->getDecString(i));
					if(srFlag == SR_NREAD){ suppress = true; }
					else{ suppress = false; }
				}
				////////////////////////////////////////////////////////////////////////////////
					
				 
				if( chStatus != CH_TIMEOUT && chStatus != CH_DISABLED && !suppress && chStatus !=CH_SUPPRESS){

					
					//Go to the begining of the block ///////////////////////////////////////////////////////////////////////
					increment(1," (while trying to create a TOWERHEADER Block for channel "+parser_->getDecString(i)+" !)" );
					/////////////////////////////////////////////////////////////////////////////////////////////////////////
					
					
					// Instantiate a new tower block//////////////////////////////////////////////////////////////////////////
					wToEnd = numbBytes/4-wordCounter_-1;
					DCCTBTowerBlock * towerBlock = new DCCTBTowerBlock(this,parser_,dataP_,TOWERHEADER_SIZE,wToEnd,wordCounter_,i); 
					towerBlocks_.push_back (towerBlock);
					towerBlock->parseXtalData();
					//////////////////////////////////////////////////////////////////////////////////////////////////////////
					
					
					//go to the end of the block ///////////////////////////////
					increment((towerBlock->getDataField("BLOCK LENGTH"))*2 - 1);
					////////////////////////////////////////////////////////////
						
				}
			}
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			

			// go to the begining of the block ////////////////////////////////////////////////////////////////////			
			increment(1," (while trying to create a DCC TRAILER Block !)");
			wToEnd = numbBytes/4-wordCounter_-1;
			dccTrailerBlock_ = new DCCTBTrailerBlock(parser_,dataP_,TRAILER_SIZE,wToEnd,wordCounter_,blockSize_/8,0);
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			
		}
	
	}catch( ECALTBParserException & e){}
	catch( ECALTBParserBlockException & e){
	  // uint32_t nEv = (parser_->dccEvents()).size() +1;
	  errorString_ += std::string(e.what());
	  blockError_=true;
	  //std::cout<<"cout"<<e.what();
	}
	
	

} 



DCCTBEventBlock::~DCCTBEventBlock(){
	
	std::vector<DCCTBTCCBlock *>::iterator it1;
	for(it1=tccBlocks_.begin();it1!=tccBlocks_.end();it1++){ delete (*it1);}
	tccBlocks_.clear();
	
	std::vector<DCCTBTowerBlock *>::iterator it2;
	for(it2=towerBlocks_.begin();it2!=towerBlocks_.end();it2++){ delete (*it2);}
	towerBlocks_.clear();
	
	if(srpBlock_ !=        0 ) { delete srpBlock_;       }
	if(dccTrailerBlock_ != 0 ) { delete dccTrailerBlock_;}
	
}



void DCCTBEventBlock::dataCheck(){
	
	
	std::string checkErrors("");
	
	
	// Check BOE field/////////////////////////////////////////////////////
	std::pair<bool,std::string> res =  checkDataField("BOE",BOE);
	if(!res.first){ checkErrors += res.second; (errors_["DCC::HEADER"])++; }
	///////////////////////////////////////////////////////////////////////
	
	
	// Check H Field //////////////////////////////////////////////////////
	std::string hField= std::string("H");
	res = checkDataField(hField,1);
	if(!res.first){ checkErrors += res.second; (errors_["DCC::HEADER"])++; }
	////////////////////////////////////////////////////////////////////////
	
	
	// Check Headers //////////////////////////////////////////////////////////
	uint32_t dccHeaderWords = 0;
	
	if(emptyEvent){ dccHeaderWords = 2;}
	else if(!emptyEvent){ dccHeaderWords = 7;}

	for(uint32_t i = 1; i<=dccHeaderWords ; i++){

		std::string header= std::string("H") + parser_->getDecString(i);
		res = checkDataField(header,i);
		if(!res.first){ checkErrors += res.second; (errors_["DCC::HEADER"])++; }
	}
	////////////////////////////////////////////////////////////////////////////
	
	
	// Check event length ///////////////////////////////////////////////////////
	res = checkDataField("EVENT LENGTH",blockSize_/8);
	if(!res.first){ checkErrors += res.second; (errors_["DCC::EVENT LENGTH"])++; }
	/////////////////////////////////////////////////////////////////////////////
		
	
	if(checkErrors!=""){
		errorString_ +="\n ======================================================================\n"; 		
		errorString_ += std::string(" ") + name_ + std::string(" data fields checks errors : ") ;
		errorString_ += checkErrors ;
		errorString_ += "\n ======================================================================";
		blockError_ = true;	
	}
	
	
}



void  DCCTBEventBlock::displayEvent(std::ostream & os){

  os << "\n\n\n\n\n >>>>>>>>>>>>>>>>>>>> Event started at word position " << std::dec << wordBufferOffset_ <<" <<<<<<<<<<<<<<<<<<<<"<<std::endl;
  	
	// Display DCC Header ///
	displayData(os);
	/////////////////////////
		
		
	// Display SRP Block Contents //////////////
	if(srpBlock_){ srpBlock_->displayData(os);}
	////////////////////////////////////////////
		
		
	// Display TCC Block Contents ///////////////////////////////
	std::vector<DCCTBTCCBlock *>::iterator it1;
	for( it1 = tccBlocks_.begin(); it1!= tccBlocks_.end(); it1++){
		(*it1)->displayData(os);
	}
	
	// Display Towers Blocks /////////////////////////////////////
	std::vector<DCCTBTowerBlock *>::iterator it2;
	for(it2 = towerBlocks_.begin();it2!=towerBlocks_.end();it2++){
			
		(*it2)->displayData(os);
			
		// Display Xtal Data /////////////////////////////////////
		std::vector<DCCTBXtalBlock * > &xtalBlocks = (*it2)->xtalBlocks();
		std::vector<DCCTBXtalBlock * >::iterator it3;
		for(it3 = xtalBlocks.begin();it3!=xtalBlocks.end();it3++){
			(*it3)->displayData(os);
		}	
	}
		
	// Display Trailer Block Contents /////////////////////////
	if(dccTrailerBlock_){ dccTrailerBlock_->displayData(os);}
		
}


std::pair<bool,std::string> DCCTBEventBlock::compare(DCCTBEventBlock * block){
	
	// DCC Header comparision /////////////////////////////// 
	std::pair<bool,std::string> ret(DCCTBBlockPrototype::compare(block));
	/////////////////////////////////////////////////////////
	
	  std::stringstream out;
	
	// Selective readout processor block comparision ////////////////////////////////////////////
	if( srpBlock_ && block->srpBlock() ){ 
		std::pair<bool,std::string> temp( srpBlock_->compare(block->srpBlock()));
		ret.first   = ret.first & temp.first;
		out<<temp.second; 
	}else if( !srpBlock_ && block->srpBlock() ){
		ret.first   = false;	
		out<<"\n ====================================================================="
		   <<"\n ERROR SR block identified in the ORIGINAL BLOCK ... "
		   <<"\n ... but the block is not present in the COMPARISION BLOCK !" 
		   <<"\n =====================================================================";
	}else if( srpBlock_ && !(block->srpBlock()) ){
		ret.first   = false;	
		out<<"\n ====================================================================="
		   <<"\n ERROR SR block identified in the COMPARISION BLOCK ... "
		   <<"\n ... but the block is not present in the ORIGINAL BLOCK !" 
		   <<"\n =====================================================================";
	}
	////////////////////////////////////////////////////////////////////////////////////////////
	
	
	
	// TCC Blocks comparision ////////////////////////////////////////////////////////
	// check number of TCC blocks 
	int numbTccBlocks_a = tccBlocks_.size();
	int numbTccBlocks_b = block->tccBlocks().size();
	
	if( numbTccBlocks_a != numbTccBlocks_b ){
		ret.first = false;
		out<<"\n ====================================================================="
		   <<"\n ERROR number of TCC blocks in the ORIGINAL BLOCK( ="<<numbTccBlocks_a<<" )"
		   <<"\n and in the COMPARISION BLOCK( = "<<numbTccBlocks_b<<" is different !"
		   <<"\n =====================================================================";
	}
	
	std::vector<DCCTBTCCBlock *>::iterator it1Tcc    = tccBlocks_.begin();
	std::vector<DCCTBTCCBlock *>::iterator it1TccEnd = tccBlocks_.end();
	std::vector<DCCTBTCCBlock *>::iterator it2Tcc    = block->tccBlocks().begin();
	std::vector<DCCTBTCCBlock *>::iterator it2TccEnd = block->tccBlocks().end();
	
	for( ; it1Tcc!=it1TccEnd && it2Tcc!=it2TccEnd; it1Tcc++, it2Tcc++){
		std::pair<bool,std::string> temp( (*it1Tcc)->compare( *it2Tcc ) );
		ret.first   = ret.first & temp.first;
		out<<temp.second; 
	}
	//////////////////////////////////////////////////////////////////////////////////
	
	
	// FE Blocks comparision ////////////////////////////////////////////////////////
	// check number of FE blocks 
	int numbTowerBlocks_a = towerBlocks_.size();
	int numbTowerBlocks_b = block->towerBlocks().size();
	
	if( numbTowerBlocks_a != numbTowerBlocks_b ){
		ret.first = false;
		out<<"\n ====================================================================="
		   <<"\n ERROR number of Tower blocks in the ORIGINAL BLOCK( ="<<numbTowerBlocks_a<<" )"
		   <<"\n and in the COMPARISION BLOCK( = "<<numbTowerBlocks_b<<" is different !"
		   <<"\n =====================================================================";
	}
	
	std::vector<DCCTBTowerBlock *>::iterator it1Tower    = towerBlocks_.begin();
	std::vector<DCCTBTowerBlock *>::iterator it1TowerEnd  = towerBlocks_.end();
	std::vector<DCCTBTowerBlock *>::iterator it2Tower    = (block->towerBlocks()).begin();
	std::vector<DCCTBTowerBlock *>::iterator it2TowerEnd = (block->towerBlocks()).end();
	
	for( ; it1Tower!=it1TowerEnd && it2Tower!=it2TowerEnd; it1Tower++, it2Tower++){
		
		std::pair<bool,std::string> temp( (*it1Tower)->compare( *it2Tower ) );
		ret.first   = ret.first & temp.first;
		out<<temp.second;

		// Xtal Block comparision ////////////////////////////
		std::vector<DCCTBXtalBlock *> xtalBlocks1( (*it1Tower)->xtalBlocks());
		std::vector<DCCTBXtalBlock *> xtalBlocks2( (*it2Tower)->xtalBlocks());
		// check number of xtal blocks 
    	int numbXtalBlocks_a = xtalBlocks1.size();
		int numbXtalBlocks_b = xtalBlocks2.size();
	
		if( numbXtalBlocks_a != numbXtalBlocks_b ){
			ret.first = false;
			out<<"\n ====================================================================="
			   <<"\n ERROR number of Xtal blocks in this TOWER ORIGINAL BLOCK( ="<<numbXtalBlocks_a<<" )"
		  	   <<"\n and in the TOWER COMPARISION BLOCK( = "<<numbXtalBlocks_b<<" is different !"
		  	   <<"\n =====================================================================";
		}
		
		std::vector<DCCTBXtalBlock *>::iterator it1Xtal    = xtalBlocks1.begin();
		std::vector<DCCTBXtalBlock *>::iterator it1XtalEnd = xtalBlocks1.end();
		std::vector<DCCTBXtalBlock *>::iterator it2Xtal    = xtalBlocks1.begin();
		std::vector<DCCTBXtalBlock *>::iterator it2XtalEnd = xtalBlocks2.end();
		
		for( ; it1Xtal!=it1XtalEnd && it2Xtal!=it2XtalEnd; it1Xtal++, it2Xtal++){
			std::pair<bool,std::string> temp( (*it1Xtal)->compare( *it2Xtal ) );
			ret.first   = ret.first & temp.first;
			out<<temp.second; 
		}
		
	}
	
	
	// Trailer block comparision ////////////////////////////////////////////
	if(  block->trailerBlock() && trailerBlock() ){ 
		std::pair<bool,std::string> temp( trailerBlock()->compare(block->trailerBlock()));
		ret.first   = ret.first & temp.first;
		out<<temp.second; 
	}
	
	ret.second += out.str(); 

	return ret;
}
			
		
		
		
		
		
		
		

bool DCCTBEventBlock::eventHasErrors(){
	
	bool ret(false);
	ret = blockError() ;
	

	// See if we have errors in the  TCC Block Contents ///////////////////////////////
	std::vector<DCCTBTCCBlock *>::iterator it1;
	for( it1 = tccBlocks_.begin(); it1!= tccBlocks_.end(); it1++){
		ret |= (*it1)->blockError();
	}
	/////////////////////////////////////////////////////////////		

	// See if we have errors in the SRP Block /////////
	if(srpBlock_){  ret |= srpBlock_->blockError(); }
	////////////////////////////////////////////////////

	
	// See if we have errors in the Trigger Blocks ///////////////
	std::vector<DCCTBTowerBlock *>::iterator it2;
	for(it2 = towerBlocks_.begin();it2!=towerBlocks_.end();it2++){
			
		ret |= (*it2)->blockError();
			
		// See if we have errors in the Xtal Data /////////////////
		std::vector<DCCTBXtalBlock * > & xtalBlocks = (*it2)->xtalBlocks();
		std::vector<DCCTBXtalBlock * >::iterator it3;
		for(it3 = xtalBlocks.begin();it3!=xtalBlocks.end();it3++){
			ret |= (*it3)->blockError();
		}
		////////////////////////////////////////////////////////////
	}
	///////////////////////////////////////////////////////////////
	
		
	// See if we have errors in the  trailler ///////////////////
	if(dccTrailerBlock_){ ret |= dccTrailerBlock_->blockError();}
	/////////////////////////////////////////////////////////////
	
	
	return ret;
	
}


std::string DCCTBEventBlock::eventErrorString(){
	
	std::string ret("");
	
	if( eventHasErrors() ){
		
		
		ret +="\n ======================================================================\n"; 		
		ret += std::string(" Event Erros occurred for L1A ( decoded value ) = ") ; 
		ret += parser_->getDecString(getDataField("LV1"));
		ret += "\n ======================================================================";
		
	
		ret += errorString();
	
		// TODO ::
		// See if we have errors in the  TCC Block Contents /////////////
		std::vector<DCCTBTCCBlock *>::iterator it1;
		for( it1 = tccBlocks_.begin(); it1!= tccBlocks_.end(); it1++){
			ret += (*it1)->errorString();
		}
		/////////////////////////////////////////////////////////////////
		// See if we have errors in the SRP Block ////
		 if(srpBlock_){  ret += srpBlock_->errorString(); }
		/////////////////////////////////////////////////////////////////
	
	
		// See if we have errors in the Tower Blocks ///////////////////////////////////////////////////
		std::vector<DCCTBTowerBlock *>::iterator it2;
		
		for(it2 = towerBlocks_.begin();it2!=towerBlocks_.end();it2++){
			
			ret += (*it2)->errorString();
			
			// See if we have errors in the Xtal Data /////////////////
			std::vector<DCCTBXtalBlock * > & xtalBlocks = (*it2)->xtalBlocks();
			std::vector<DCCTBXtalBlock * >::iterator it3;
		
		
			std::string temp;
			for(it3 = xtalBlocks.begin();it3!=xtalBlocks.end();it3++){ temp += (*it3)->errorString();}
		
			if(temp!=""){
				ret += "\n Fine grain data Errors found ...";
				ret += "\n(  Tower ID = " + parser_->getDecString( (*it2)->getDataField("TT/SC ID"));
				ret += ", LV1 = " + parser_->getDecString( (*it2)->getDataField("LV1"));
				ret += ", BX = " + parser_->getDecString( (*it2)->getDataField("BX")) + " )";
				ret += temp;
			}
			////////////////////////////////////////////////////////////
			
		}
	
		
		// See if we have errors in the  trailler ////////////////////
		if(dccTrailerBlock_){ ret += dccTrailerBlock_->errorString();}
		//////////////////////////////////////////////////////////////
	
	}
	
	return ret;
	
}




std::vector< DCCTBTowerBlock * > DCCTBEventBlock::towerBlocksById(uint32_t towerId){
	std::vector<DCCTBTowerBlock *> myVector;	
	std::vector<DCCTBTowerBlock *>::iterator it;
	
	for( it = towerBlocks_.begin(); it!= towerBlocks_.end(); it++ ){
		try{
			
			std::pair<bool,std::string> idCheck   = (*it)->checkDataField("TT/SC ID",towerId);
			
			if(idCheck.first ){ myVector.push_back( (*it) ); }
		}catch (ECALTBParserBlockException &e){/*ignore*/}
	}
	
	return myVector;
}
