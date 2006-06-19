#include "EventFilter/EcalRawToDigi/src/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCBlockPrototype.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataParser.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataMapper.h"
#include "EventFilter/EcalRawToDigi/src/DCCTowerBlock.h"
#include "ECALParserException.h"
#include "ECALParserBlockException.h"
#include "EventFilter/EcalRawToDigi/src/DCCSRPBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCTCCBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCXtalBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCTrailerBlock.h"



DCCEventBlock::DCCEventBlock(
	DCCDataParser * parser, 
	ulong * buffer, 
	ulong numbBytes, 
	ulong wordsToEnd, 
	ulong wordBufferOffset , 
	ulong wordEventOffset 
) : 
DCCBlockPrototype(parser,"DCCHEADER", buffer, numbBytes,wordsToEnd)
,dccTrailerBlock_(0),srpBlock_(0),wordBufferOffset_(wordBufferOffset) {
	
	
	//Reset error counters ////
	errors_["DCC::HEADER"] = 0;
	errors_["DCC::EVENT LENGTH"] = 0;
	///////////////////////////
	
	ulong wToEnd(0);
	
	try{ 
	
		// Get data fields from the mapper and retrieve data ///	
		if(numbBytes == DCCDataParser::EMPTYEVENTSIZE ){
			mapperFields_ = parser_->mapper()->emptyEventFields();
			emptyEvent = true;
		}else{
			mapperFields_ = parser_->mapper()->dccFields();	
			emptyEvent = false;
		}
		
		try{ parseData(); }
		catch (ECALParserBlockException &e){/*ignore*/}
		///////////////////////////////////////////////////////
		

	
		// Check internal data //////////////
		if(parser_->debug()){ dataCheck(); }
		/////////////////////////////////////
	
	
		// Check if empty event was produced /////////////////////////////////////////////////////////////////

		if( !emptyEvent && getDataField("DCC ERRORS")!= DCCERROR_EMPTYEVENT ){
			 		
			// Build the SRP block ////////////////////////////////////////////////////////////////////////////////////
		
			bool srp(false);
			ulong sr_ch = getDataField("SR_CHSTATUS");
			if( sr_ch!=CH_TIMEOUT  && sr_ch != CH_DISABLED ){ 			
				
				//Go to the begining of the block
				increment(1," (while trying to create a SR Block !)");
				wToEnd = numbBytes/4-wordCounter_-1;	
				
				// Build SRP Block //////////////////////////////////////////////////////////////////////
				srpBlock_ = new DCCSRPBlock( this, parser_, dataP_, parser_->srpBlockSize(), wToEnd,wordCounter_);
				//////////////////////////////////////////////////////////////////////////////////////////
		
				increment((parser_->srpBlockSize())/4-1);
				if(getDataField("SR")){ srp=true; }
			}	
			
			////////////////////////////////////////////////////////////////////////////////////////////////////////////		


			// Build TCC blocks ////////////////////////////////////////////////////////////////////////////////////////
			for(ulong i=1; i<=4;i++){
				ulong tcc_ch, tccId(0);
				if( i == 1){ tccId = parser_->tcc1Id();}
				if( i == 2){ tccId = parser_->tcc2Id();}
				if( i == 3){ tccId = parser_->tcc3Id();}	
				if( i == 4){ tccId = parser_->tcc4Id();}
				
				string tcc = string("TCC_CHSTATUS#") + parser_->getDecString(i);	
				tcc_ch = getDataField(tcc);
				
				if( tcc_ch != CH_TIMEOUT && tcc_ch != CH_DISABLED){	 
					// Go to the begining of the block
					increment(1," (while trying to create a"+tcc+" Block !)");
					
					wordsToEnd = numbBytes/4-wordCounter_-1;	
					// Build TCC Block /////////////////////////////////////////////////////////////////////////////////
					tccBlocks_.push_back(  new DCCTCCBlock( this, parser_, dataP_,parser_->tccBlockSize(), wToEnd,wordCounter_, tccId));
					//////////////////////////////////////////////////////////////////////////////////////////////////////	
					
					increment((parser_->tccBlockSize())/4-1);
				}
			}
			////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
			
			
			// Build channel data //////////////////////////////////////////////////////////////////////////////////////////////////////	
			// See number of channels that we need according to the trigger type //
			// TODO : WHEN IN LOCAL MODE WE SHOULD CHECK RUN TYPE
			ulong triggerType = getDataField("TRIGGER TYPE");			
			ulong numbChannels;
			if( triggerType == PHYSICTRIGGER )          { numbChannels = 68; }
			else if (triggerType == CALIBRATIONTRIGGER ){ numbChannels = 70; }
			// TODO :: implement other triggers
			else{
				string error = string("\n DCC::HEADER TRIGGER TYPE = ")+parser_->getDecString(triggerType)+string(" is not a valid type !");
				ECALParserBlockException a(error);
				throw a;
			}
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
			
			
			//ulong chStatus, srFlag;
			ulong srFlag;
			bool suppress(false);
			
			for( ulong i=1; i<=numbChannels; i++){
				
				string chStatusId = string("FE_CHSTATUS#") + parser_->getDecString(i);
				ulong chStatus = getDataField(chStatusId);
				
				// If srp is on, we need to check if channel was suppressed ////////////////////
				if(srp){ 
					srFlag   = srpBlock_->getDataField( string("SR#") + parser_->getDecString(i));
					if(srFlag == SR_NREAD){ suppress = true; }
					else{ suppress = false; }
				}
				////////////////////////////////////////////////////////////////////////////////
					
				 
				if( chStatus != CH_TIMEOUT && chStatus != CH_DISABLED && !suppress){

					
					//Go to the begining of the block ///////////////////////////////////////////////////////////////////////
					increment(1," (while trying to create a TOWERHEADER Block for channel "+parser_->getDecString(i)+" !)" );
					/////////////////////////////////////////////////////////////////////////////////////////////////////////
					
					
					// Instantiate a new tower block//////////////////////////////////////////////////////////////////////////
					wToEnd = numbBytes/4-wordCounter_-1;
					DCCTowerBlock * towerBlock = new DCCTowerBlock(this,parser_,dataP_,TOWERHEADER_SIZE,wToEnd,wordCounter_,i); 
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
			dccTrailerBlock_ = new DCCTrailerBlock(parser_,dataP_,TRAILER_SIZE,wToEnd,wordCounter_,blockSize_/8,0);
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			
		}
	
	}catch( ECALParserException & e){}
	catch( ECALParserBlockException & e){
	        //ulong nEv = (parser_->dccEvents()).size() +1;
		errorString_ += string(e.what());
		blockError_=true;
		//cout<<"cout"<<e.what();
	}
	
	

} 



DCCEventBlock::~DCCEventBlock(){
	
	vector<DCCTCCBlock *>::iterator it1;
	for(it1=tccBlocks_.begin();it1!=tccBlocks_.end();it1++){ delete (*it1);}
	tccBlocks_.clear();
	
	vector<DCCTowerBlock *>::iterator it2;
	for(it2=towerBlocks_.begin();it2!=towerBlocks_.end();it2++){ delete (*it2);}
	towerBlocks_.clear();
	
	if(srpBlock_ !=        0 ) { delete srpBlock_;       }
	if(dccTrailerBlock_ != 0 ) { delete dccTrailerBlock_;}
	
}



void DCCEventBlock::dataCheck(){
	
	
	string checkErrors("");
	
	
	// Check BOE field/////////////////////////////////////////////////////
	pair<bool,string> res =  checkDataField("BOE",BOE);
	if(!res.first){ checkErrors += res.second; (errors_["DCC::HEADER"])++; }
	///////////////////////////////////////////////////////////////////////
	
	
	// Check H Field //////////////////////////////////////////////////////
	string hField= string("H");
	res = checkDataField(hField,1);
	if(!res.first){ checkErrors += res.second; (errors_["DCC::HEADER"])++; }
	////////////////////////////////////////////////////////////////////////
	
	
	// Check Headers //////////////////////////////////////////////////////////
	ulong dccHeaderWords;
	
	if(emptyEvent){ dccHeaderWords = 2;}
	else if(!emptyEvent){ dccHeaderWords = 7;}

	for(ulong i = 1; i<=dccHeaderWords ; i++){

		string header= string("H") + parser_->getDecString(i);
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
		errorString_ += string(" ") + name_ + string(" data fields checks errors : ") ;
		errorString_ += checkErrors ;
		errorString_ += "\n ======================================================================";
		blockError_ = true;	
	}
	
	
}



void  DCCEventBlock::displayEvent(ostream & os){

  	os<<"\n Event started at word position "<<dec<<wordBufferOffset_<<endl;
  	
	// Display DCC Header ///
	displayData(os);
	/////////////////////////
		
		
	// Display SRP Block Contents //////////////
	if(srpBlock_){ srpBlock_->displayData(os);}
	////////////////////////////////////////////
		
		
	// Display TCC Block Contents ///////////////////////////////
	vector<DCCTCCBlock *>::iterator it1;
	for( it1 = tccBlocks_.begin(); it1!= tccBlocks_.end(); it1++){
		(*it1)->displayData(os);
	}
	/////////////////////////////////////////////////////////////
		
		
	
	// Display Towers Blocks /////////////////////////////////////
	vector<DCCTowerBlock *>::iterator it2;
	for(it2 = towerBlocks_.begin();it2!=towerBlocks_.end();it2++){
			
		(*it2)->displayData(os);
			
		// Display Xtal Data /////////////////////////////////////
		vector<DCCXtalBlock * > & xtalBlocks = (*it2)->xtalBlocks();
		vector<DCCXtalBlock * >::iterator it3;
		for(it3 = xtalBlocks.begin();it3!=xtalBlocks.end();it3++){
			(*it3)->displayData(os);
		}
		////////////////////////////////////////////////////////////
			
	}
	///////////////////////////////////////////////////////////////
	
		
	// Display Trailer Block Contents /////////////////////////
	if(dccTrailerBlock_){ dccTrailerBlock_->displayData(os);}
	///////////////////////////////////////////////////////////
		
}





bool DCCEventBlock::eventHasErrors(){
	
	bool ret(false);
	ret = blockError() ;
	

	// See if we have errors in the  TCC Block Contents ///////////////////////////////
	vector<DCCTCCBlock *>::iterator it1;
	for( it1 = tccBlocks_.begin(); it1!= tccBlocks_.end(); it1++){
		ret |= (*it1)->blockError();
	}
	/////////////////////////////////////////////////////////////		

	// See if we have errors in the SRP Block /////////
	if(srpBlock_){  ret |= srpBlock_->blockError(); }
	////////////////////////////////////////////////////

	
	// See if we have errors in the Trigger Blocks ///////////////
	vector<DCCTowerBlock *>::iterator it2;
	for(it2 = towerBlocks_.begin();it2!=towerBlocks_.end();it2++){
			
		ret |= (*it2)->blockError();
			
		// See if we have errors in the Xtal Data /////////////////
		vector<DCCXtalBlock * > & xtalBlocks = (*it2)->xtalBlocks();
		vector<DCCXtalBlock * >::iterator it3;
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


string DCCEventBlock::eventErrorString(){
	
	string ret("");
	
	if( eventHasErrors() ){
		
		
		ret +="\n ======================================================================\n"; 		
		ret += string(" Event Erros occurred for LV1 accept ( decoded value ) = ") ; 
		ret += parser_->getDecString(getDataField("LV1"));
		ret += "\n ======================================================================";
		
	
		ret += errorString();
	
		// TODO ::
		// See if we have errors in the  TCC Block Contents /////////////
		vector<DCCTCCBlock *>::iterator it1;
		for( it1 = tccBlocks_.begin(); it1!= tccBlocks_.end(); it1++){
			ret += (*it1)->errorString();
		}
		/////////////////////////////////////////////////////////////////
		// See if we have errors in the SRP Block ////
		 if(srpBlock_){  ret += srpBlock_->errorString(); }
		/////////////////////////////////////////////////////////////////
	
	
		// See if we have errors in the Tower Blocks ///////////////////////////////////////////////////
		vector<DCCTowerBlock *>::iterator it2;
		
		for(it2 = towerBlocks_.begin();it2!=towerBlocks_.end();it2++){
			
			ret += (*it2)->errorString();
			
			// See if we have errors in the Xtal Data /////////////////
			vector<DCCXtalBlock * > & xtalBlocks = (*it2)->xtalBlocks();
			vector<DCCXtalBlock * >::iterator it3;
		
		
			string temp;
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




vector< DCCTowerBlock * > DCCEventBlock::towerBlocksById(ulong towerId){
	vector<DCCTowerBlock *> myVector;	
	vector<DCCTowerBlock *>::iterator it;
	
	for( it = towerBlocks_.begin(); it!= towerBlocks_.end(); it++ ){
		try{
			
			pair<bool,string> idCheck   = (*it)->checkDataField("TT/SC ID",towerId);
			
			if(idCheck.first ){ myVector.push_back( (*it) ); }
		}catch (ECALParserBlockException &e){/*ignore*/}
	}
	
	return myVector;
}
