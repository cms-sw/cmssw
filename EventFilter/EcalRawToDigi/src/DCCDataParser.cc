
#include "EventFilter/EcalRawToDigi/src/DCCDataParser.h"
#include "ECALParserException.h"
#include "EventFilter/EcalRawToDigi/src/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataMapper.h"

DCCDataParser::DCCDataParser(bool parseInternalData,bool debug):
buffer_(0),parseInternalData_(parseInternalData),debug_(debug) {

	 mapper_ = new DCCDataMapper(this);
	 resetErrorCounters();
	 computeBlockSizes();

}

DCCDataParser::DCCDataParser(vector<ulong> parserParameters, bool parseInternalData,bool debug):
buffer_(0),parseInternalData_(parseInternalData),debug_(debug), parameters(parserParameters){

  LogDebug("EcalRawToDigi") << "@SUB=DCCDataParser";

	 mapper_ = new DCCDataMapper(this);
	 resetErrorCounters();
	 computeBlockSizes();

}


DCCDataParser::~DCCDataParser(){
	
	
	// delete DCCEvents if any... ////////////////////////////////////
	
	vector<DCCEventBlock *>::iterator it;
	for(it=dccEvents_.begin();it!=dccEvents_.end();it++){delete *it;}
	dccEvents_.clear();
	//////////////////////////////////////////////////////////////////
	
	delete 	mapper_;
	
	
}



void DCCDataParser::computeBlockSizes(){
	ulong nTT = numbTTs();
	ulong tSamples = numbTriggerSamples();
	ulong nSr = numbSRF();
	ulong tf(0), srf(0);
	
	if( ( (nTT*tSamples) < 4 ) || ( nTT*tSamples )%4 ){tf=1;}
	else{tf=0;}
	
	if( (srf<16) || srf%16 ){ srf=1;}
	else{srf=0;}
	
	tccBlockSize_ = 8 + ( (nTT*tSamples)/4 )*8 + tf*8 ;
	srpBlockSize_ = 8 + ( nSr/16 )*8 + srf*8;


}

void DCCDataParser::parseFile(string fileName, bool singleEvent ){
	
	resetErrorCounters();
	
	//Get DCC data from file /////////////////////////////////////////////////////////
	ifstream inputFile;
	inputFile.open(fileName.c_str());
	
	
	if(! inputFile.fail() ){ 
	
		string myWord;
		vector<string> dataVector;
		
		while( inputFile >> myWord ){ dataVector.push_back( myWord ); }
		
		bufferSize_ = (dataVector.size() ) * 4 ; 
		if( buffer_ ){ delete [] buffer_; }
		buffer_ = new ulong[dataVector.size()];
	
		ulong * myData_ = (ulong *) buffer_;

		//fill buffer data with data from file lines ///////////////////////////////
		for(ulong i = 1; i <= dataVector.size() ; i++, myData_++ ){
			sscanf( (dataVector[i-1]).c_str(),"%lx",myData_);
			//cout<<"\n data position"<<dec<<i<<" val = "<<getHexString(*myData_);
		}
		///////////////////////////////////////////////////////////////////////////
		
  		inputFile.close();
  		
  		try { parseBuffer( buffer_, bufferSize_, singleEvent ); }
  		catch (ECALParserException &e){
			throw 1;
		}
	}else{ 
		string errorMessage = string(" Error::Unable to open file :") + fileName;
		throw ECALParserException(errorMessage);
	}
	///////////////////////////////////////////////////////////////////////////////////////////
	
	
}




void DCCDataParser::parseBuffer(ulong * buffer, ulong bufferSize, bool singleEvent){
	
  LogDebug("EcalRawToDigi") << "@SUB=DCCDataParser::parseBuffer";

	resetErrorCounters();
	
	buffer_ = buffer;
        
	// Clear stored data //////////////////////////////////////////////

	processedEvent_ = 0;
	events_.clear();
	vector<DCCEventBlock *>::iterator it;
	for( it = dccEvents_.begin(); it!=dccEvents_.end(); it++ ){delete *it;}
	dccEvents_.clear();
	eventErrors_ = "";

	//////////////////////////////////////////////////////////////////

	LogDebug("EcalRawToDigi") << "DCCDataParser:parseBuffer: Buffer Size :" << bufferSize;

	// Check if we have a coherent buffer size ///////////////////////////////////////////////////////////////////////////////
	if( bufferSize%8  ){
		string fatalError ;
		fatalError +="\n ======================================================================"; 		
		fatalError +="\n Fatal error at event = " + getDecString(events_.size()+1);
		fatalError +="\n Buffer Size of = "+ getDecString(bufferSize) + "[bytes] is not divisible by 8 ... ";
		fatalError +="\n ======================================================================";
		throw ECALParserException(fatalError);
	}
	if ( bufferSize < EMPTYEVENTSIZE ){
		string fatalError ;
		fatalError +="\n ======================================================================"; 		
		fatalError +="\n Fatal error at event = " + getDecString(events_.size()+1);
		fatalError +="\n Buffer Size of = "+ getDecString(bufferSize) + "[bytes] is less than an empty event ... ";
		fatalError +="\n ======================================================================";
		throw ECALParserException(fatalError);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	ulong * myPointer =  buffer_;

	//ulong processedBytes(0), wordIndex(0), lastEvIndex(0),eventSize(0), eventLength(0), errorMask(0) ;
	ulong processedBytes(0), wordIndex(0), eventLength(0), errorMask(0) ;

	while( processedBytes + EMPTYEVENTSIZE <= bufferSize ){

		//cout<<endl;
		//cout<<"-> EventSizeBytes   =   "<<dec<<bufferSize<<endl;
		//cout<<"-> processedBytes.  =   "<<dec<<processedBytes<<endl;
		//cout<<"-> Processed Event  =   "<<dec<<processedEvent_<<endl;
		//cout<<"-> First ev. word   = 0x"<<hex<<(*myPointer)<<endl;
		//cout<<"-> word index       =   "<<dec<<wordIndex<<endl;

		// Check if Event Length is Coherent /////////////////////////////////////////
		ulong bytesToEnd         = bufferSize - processedBytes;
		pair<ulong,ulong> eventD = checkEventLength(myPointer,bytesToEnd,singleEvent);
		eventLength              = eventD.second; 
		errorMask                = eventD.first;
		//////////////////////////////////////////////////////////////////////////////

		// Debug ////////////////////////////////////////////////////////////////////////
		//cout<<endl;
		//cout<<" out... Bytes To End.... =   "<<dec<<bytesToEnd<<endl;
		//cout<<" out... Processed Event  =   "<<dec<<processedEvent_<<endl;	
		//cout<<" out... Event Length     =   "<<dec<<eventLength<<endl;
		//cout<<" out... LastWord         = 0x"<<hex<<*(myPointer+eventLength*2-1)<<endl;
		//////////////////////////////////////////////////////////////////////////////////

		// BuildEvent and display data ////////////////////////////////////////////////////////////////////////////
		if (parseInternalData_){
			
			DCCEventBlock * myBlock = new DCCEventBlock(this,myPointer,eventLength*8, eventLength*2 -1 ,wordIndex,0);
			dccEvents_.push_back(myBlock);
			
		}
		pair<ulong *, ulong> eventPointer(myPointer,eventLength);
		pair<ulong, pair<ulong *, ulong> > eventPointerWithErrorMask(errorMask,eventPointer);
		events_.push_back(eventPointerWithErrorMask);
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		
		// update processed buffer size ///////////////////////////
		processedEvent_++;
		processedBytes += eventLength*8;
		//cout<<"\n Processed Bytes = "<<dec<<processedBytes<<endl;
		//////////////////////////////////////////////////////////
		
		
		
		// Go to next event///////////
		myPointer     += eventLength*2;
		wordIndex     += eventLength*2;
		///////////////////////////////

	}
	

}


pair<ulong,ulong> DCCDataParser::checkEventLength(ulong * pointerToEvent, ulong bytesToEnd, bool singleEvent){
	
	pair<ulong,ulong> result; //returns error mask and event length 
	
	// Returned error mask (0 = no error) [ bit 1 to BOE Error, bit 2 to LENGTH error, bit 3 to EOE Error ]
	ulong errorMask(0);

	// Check BOE /////////////////////////////////////////////////////////////////////////////
	ulong * boePointer = pointerToEvent + 1;
	if( ((*boePointer)>>BOEBEGIN)&BOEMASK != BOE ){ (errors_["DCC::BOE"])++; errorMask = 1; }
	//////////////////////////////////////////////////////////////////////////////////////////
	
	// Get Event Length from buffer /////////////////
	ulong * myPointer = pointerToEvent + 2; 
	ulong eventLength = (*myPointer)&EVENTLENGTHMASK;
	/////////////////////////////////////////////////
	
	
	//cout<<" Event Length(from decoding) = "<<dec<<eventLength<<"... bytes to end... "<<bytesToEnd<<", event numb : "<<processedEvent_<<endl;
	
	
	bool eoeError = false;
	

	if( singleEvent && eventLength != bytesToEnd/8 ){
		eventLength = bytesToEnd/8;
		(errors_["DCC::EVENT LENGTH"])++; errorMask = errorMask | (1<<1);
	}

	else if( eventLength == 0 || eventLength > (bytesToEnd / 8) || eventLength < (EMPTYEVENTSIZE/8) ){  
		
		// How to handle bad event length in multiple event buffers
		// First approach : Send an exception	
		// Second aproach : Try to find the EOE (To be done? If yes check dataDecoder tBeam implementation)
		string fatalError;
		
		fatalError +="\n ======================================================================"; 		
		fatalError +="\n Fatal error at event = " + getDecString(events_.size()+1);
		fatalError +="\n Decoded event length = " + getDecString(eventLength);
		fatalError +="\n bytes to buffer end  = " + getDecString(bytesToEnd);
		fatalError +="\n Unable to procead the data decoding ...";
		if(eventLength > (bytesToEnd / 8)){ fatalError +=" (eventLength > (bytesToEnd / 8)";}
		else{ fatalError += "\n event length not big enouph heaven to build an empty event ( 4x8 bytes)";}
		fatalError +="\n ======================================================================";
		throw ECALParserException(fatalError);
	
	}
	
	// check EOE event //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	ulong * endOfEventPointer = pointerToEvent + eventLength*2 -1;
	if ( ((*endOfEventPointer)>>EOEBEGIN & EOEMASK != EOEMASK) && !eoeError ){ (errors_["DCC::EOE"])++; errorMask = errorMask | (1<<2); }
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
	result.first  = errorMask;
	result.second = eventLength;
	
	return result;
	
}




string DCCDataParser::index(ulong position){
	
	char indexBuffer[20];
	sprintf(indexBuffer,"W[%08lu]",position);
	return string(indexBuffer);	
	
}


string DCCDataParser::getDecString(ulong data){
	
	char buffer[10];
	sprintf(buffer,"%lu",data);
	return string(buffer);	
}

string DCCDataParser::getHexString(ulong data){
	
	char buffer[10];
	sprintf(buffer,"0x%08lx",data);
	return string(buffer);	
}


string DCCDataParser::getIndexedData( ulong index, ulong * pointer){

	string ret;
	
	char indexBuffer[20];
	char dataBuffer[20];
	
	sprintf(indexBuffer,"W[%08lu] = ",index);
	sprintf(dataBuffer,"0x%08lx",*pointer);
	
	ret = string(indexBuffer)+string(dataBuffer);
	
	return ret;	
	
}

void DCCDataParser::resetErrorCounters(){
	
	errors_["DCC::BOE"]         = 0;
	errors_["DCC::EOE"]         = 0;
	errors_["DCC::EVENT LENGTH"] = 0;	
	
}

