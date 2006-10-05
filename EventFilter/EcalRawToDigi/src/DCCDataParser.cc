
#include "DCCDataParser.h"



/*----------------------------------------------*/
/* DCCDataParser::DCCDataParser                 */
/* class constructor                            */
/*----------------------------------------------*/
DCCDataParser::DCCDataParser(vector<ulong> parserParameters, bool parseInternalData,bool debug):
  buffer_(0),parseInternalData_(parseInternalData),debug_(debug), parameters(parserParameters){
	
  mapper_ = new DCCDataMapper(this);       //build a new data mapper
  resetErrorCounters();                    //restart error counters
  computeBlockSizes();                     //calculate block sizes
  
}


/*----------------------------------------------*/
/* DCCDataParser::resetErrorCounters            */
/* resets error counters                        */
/*----------------------------------------------*/
void DCCDataParser::resetErrorCounters(){
  //set error counters to 0
  errors_["DCC::BOE"] = 0;               //begin of event (header B[60-63])
  errors_["DCC::EOE"] = 0;               //end of event (trailer B[60-63])
  errors_["DCC::EVENT LENGTH"] = 0;	 //event length (trailer B[32-55])
}


/*----------------------------------------------*/
/* DCCDataParser::computeBlockSizes             */
/* calculate the size of TCC and SR blocks      */
/*----------------------------------------------*/
void DCCDataParser::computeBlockSizes(){
  ulong nTT = numbTTs();                                      //gets the number of the trigger towers (default:68)  
  ulong tSamples = numbTriggerSamples();                      //gets the number of trigger time samples (default: 1)
  ulong nSr = numbSRF();                                      //gests the number of SR flags (default:68)

  ulong tf(0), srf(0);
	
  if( (nTT*tSamples)<4 || (nTT*tSamples)%4 ) tf=1;            //test is there is no TTC primitives or if it's a multiple of 4?
  else tf=0;
  

  if( srf<16 || srf%16 ) srf=1;                               //??? by default srf=0 why do we make this test ?????
  else srf=0;
  
  //TTC block size: header (8 bytes) + 17 words with 4 trigger primitives (17*8bytes)
  tccBlockSize_ = 8 + ((nTT*tSamples)/4)*8 + tf*8 ;          

  //SR block size: header (8 bytes) + 4 words with 16 SR flags + 1 word with 4 SR flags (5*8bytes)
  srpBlockSize_ = 8 + (nSr/16)*8 + srf*8;
}


/*------------------------------------------------*/
/* DCCDataParser::parseFile                       */
/* reada data from file and parse it              */
/*------------------------------------------------*/
void DCCDataParser::parseFile(string fileName, bool singleEvent){
	
  ifstream inputFile;                            //open file as input
  inputFile.open(fileName.c_str());
	
  resetErrorCounters();                          //reset error counters

  //for debug purposes
  //cout << "Now in DCCDataParser::parseFile " << endl;

	
  //if file opened correctly read data to a buffer and parse it
  //else throw an exception
  if( !inputFile.fail() ){ 
	
    string myWord;                               //word read from line
    vector<string> dataVector;                   //data vector
    
    //until the end of file read each line as a string and add it to the data vector
    while( inputFile >> myWord ){ 
      dataVector.push_back( myWord ); 
    }
    
    bufferSize_ = (dataVector.size() ) * 4 ;      //buffer size in bytes (note:each char is an hex number)
    if( buffer_ ){ delete [] buffer_; }           //delete current vector if any
    buffer_ = new ulong[dataVector.size()];       //allocate memory for new data buffer

    ulong *myData_ = (ulong *) buffer_;         
    
    //fill buffer data with data from file lines
    for(ulong i = 1; i <= dataVector.size() ; i++, myData_++ ){
      sscanf((dataVector[i-1]).c_str(),"%x",(uint *)myData_);
      
      //for debug purposes
      //cout << endl << "Data position: " << dec << i << " val = " << getHexString(*myData_);
    }
    
    inputFile.close();                              //close file
    
    parseBuffer( buffer_,bufferSize_,singleEvent);  //parse data from buffer
    
  }
  else{ 
    string errorMessage = string(" Error::Unable to open file :") + fileName;
    throw ECALParserException(errorMessage);
  }
}


/*----------------------------------------------------------*/
/* DCCDataParser::parseBuffer                               */
/* parse data from a buffer                                 */
/*----------------------------------------------------------*/
void DCCDataParser::parseBuffer(ulong * buffer, ulong bufferSize, bool singleEvent){
	
  resetErrorCounters();                           //reset error counters
	
  buffer_ = buffer;                               //set class buffer
  
  //clear stored data
  processedEvent_ = 0;
  events_.clear();
  vector<DCCEventBlock *>::iterator it;
  for( it = dccEvents_.begin(); it!=dccEvents_.end(); it++ ) { delete *it; }
  dccEvents_.clear();
  eventErrors_ = "";
	
  //for debug purposes
  //cout << endl << "Now in DCCDataParser::parseBuffer" << endl;
  //cout << endl << "Buffer Size:" << dec << bufferSize << endl;
	
  //check if we have a coherent buffer size
  if( bufferSize%8  ){
    string fatalError ;
    fatalError += "\n ======================================================================"; 		
    fatalError += "\n Fatal error at event = " + getDecString(events_.size()+1);
    fatalError += "\n Buffer Size of = "+ getDecString(bufferSize) + "[bytes] is not divisible by 8 ... ";
    fatalError += "\n ======================================================================";
    throw ECALParserException(fatalError);
  }
  if ( bufferSize < EMPTYEVENTSIZE ){
    string fatalError ;
    fatalError += "\n ======================================================================"; 		
    fatalError += "\n Fatal error at event = " + getDecString(events_.size()+1);
    fatalError += "\n Buffer Size of = "+ getDecString(bufferSize) + "[bytes] is less than an empty event ... ";
    fatalError += "\n ======================================================================";
    throw ECALParserException(fatalError);
  }

  ulong *myPointer =  buffer_;                        
  
  //  ulong processedBytes(0), wordIndex(0), lastEvIndex(0),eventSize(0), eventLength(0), errorMask(0);
  ulong processedBytes(0), wordIndex(0), eventLength(0), errorMask(0);
  
  //parse until there are no more events
  while( processedBytes + EMPTYEVENTSIZE <= bufferSize ){

    //for debug purposes
    //cout << "-> processedBytes.  =   " << dec << processedBytes << endl;
    //cout << " -> Processed Event index =   " << dec << processedEvent_ << endl;
    //cout << "-> First ev.word    = 0x" << hex << (*myPointer) << endl;
    //cout << "-> word index       =   " << dec << wordIndex << endl;
    
    //check if Event Length is coherent /////////////////////////////////////////
    ulong bytesToEnd         = bufferSize - processedBytes;
    pair<ulong,ulong> eventD = checkEventLength(myPointer,bytesToEnd,singleEvent);
    eventLength              = eventD.second; 
    errorMask                = eventD.first;
    //////////////////////////////////////////////////////////////////////////////
     
    //for debug purposes
    //cout <<" -> EventSizeBytes        =   " << dec << eventLength*8 << endl;
   
	  
	     
    //for debug purposes debug 
    //cout<<endl;
    //cout<<" out... Bytes To End.... =   "<<dec<<bytesToEnd<<endl;
    //cout<<" out... Processed Event  =   "<<dec<<processedEvent_<<endl;	
    //cout<<" out... Event Length     =   "<<dec<<eventLength<<endl;
    //cout<<" out... LastWord         = 0x"<<hex<<*(myPointer+eventLength*2-1)<<endl;
    
    if (parseInternalData_){ 
      //build a new event block from buffer
      DCCEventBlock *myBlock = new DCCEventBlock(this,myPointer,eventLength*8, eventLength*2 -1 ,wordIndex,0);
      
      //add event to dccEvents vector
      dccEvents_.push_back(myBlock);
    }
    
    //build the event pointer with error mask and add it to the events vector
    pair<ulong *, ulong> eventPointer(myPointer,eventLength);
    pair<ulong, pair<ulong*, ulong> > eventPointerWithErrorMask(errorMask,eventPointer);
    events_.push_back(eventPointerWithErrorMask);
    		
    //update processed buffer size 
    processedEvent_++;
    processedBytes += eventLength*8;
    //cout << endl << "Processed Bytes = " << dec << processedBytes << endl;
    
    //go to next event
    myPointer     += eventLength*2;
    wordIndex     += eventLength*2;
  } 
}


/*---------------------------------------------*/
/* DCCDataParser::checkEventLength             */
/* check if event length is consistent with    */
/* the words written in buffer                 */
/* returns a 3 bit error mask codified as:     */
/*   bit 1 - BOE error                         */
/*   bit 2 - EVENT LENGTH error                */
/*   bit 3 - EOE Error                         */
/* and the event length                        */
/*---------------------------------------------*/
pair<ulong,ulong> DCCDataParser::checkEventLength(ulong *pointerToEvent, ulong bytesToEnd, bool singleEvent){
	
  pair<ulong,ulong> result;    //returns error mask and event length 
  ulong errorMask(0);          //error mask to return

  //check begin of event (BOE bits field) 
  //(Note: we have to add one to read the 2nd 32 bit word where BOE is written)
  ulong *boePointer = pointerToEvent + 1;
  if( ((*boePointer)>>BOEBEGIN)& BOEMASK != BOE ) { 
    (errors_["DCC::BOE"])++; errorMask = 1; 
  }
	
	
  //get Event Length from buffer (Note: we have to add two to read the 3rd 32 bit word where EVENT LENGTH is written)
  ulong * myPointer = pointerToEvent + 2; 
  ulong eventLength = (*myPointer)&EVENTLENGTHMASK;

  // cout << " Event Length(from decoding) = " << dec << eventLength << "... bytes to end... " << bytesToEnd << ", event numb : " << processedEvent_ << endl;
  
  bool eoeError = false;

  //check if event is empty but but EVENT LENGTH is not corresponding to it
  if( singleEvent && eventLength != bytesToEnd/8 ){
    eventLength = bytesToEnd/8;
    (errors_["DCC::EVENT LENGTH"])++; 
    errorMask = errorMask | (1<<1);
  }
  //check if event length mismatches the number of words written as data
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
    else{ fatalError += "\n event length not big enough heaven to build an empty event ( 4x8 bytes)";}

    fatalError +="\n ======================================================================";

    throw ECALParserException(fatalError);
  }

  //check end of event (EOE bits field) 
  //(Note: event length is multiplied by 2 because its written as 32 bit words and not 64 bit words)
  ulong *endOfEventPointer = pointerToEvent + eventLength*2 -1;
  if (((*endOfEventPointer) >> EOEBEGIN & EOEMASK != EOEMASK) && !eoeError ){ 
    (errors_["DCC::EOE"])++; 
    errorMask = errorMask | (1<<2); 
  }
  
  //build result to return
  result.first  = errorMask;
  result.second = eventLength;
  
  return result;
}



/*----------------------------------------------*/
/* DCCDataParser::index                         */
/* build an index string                        */
/*----------------------------------------------*/
string DCCDataParser::index(ulong position){

  char indexBuffer[20];
  sprintf(indexBuffer,"W[%08lu]",position);        //build an index string for display purposes, p.e.  W[15] 

  return string(indexBuffer);	
}


/*-----------------------------------------------*/
/* DCCDataParser::getDecString                   */
/* print decimal data to a string                */
/*-----------------------------------------------*/
string DCCDataParser::getDecString(ulong data){
	
  char buffer[10];
  sprintf(buffer,"%lu",data);

  return string(buffer);	
}


/*-------------------------------------------------*/
/* DCCDataParser::getHexString                     */
/* print data in hexadecimal base to a string      */
/*-------------------------------------------------*/
string DCCDataParser::getHexString(ulong data){
	
  char buffer[10];
  sprintf(buffer,"0x%08x",(uint)(data));

  return string(buffer);	
}


/*------------------------------------------------*/
/* DCCDataParser::getIndexedData                  */
/* build a string with index and data             */
/*------------------------------------------------*/
string DCCDataParser::getIndexedData(ulong position, ulong *pointer){
  string ret;
  
  //char indexBuffer[20];
  //char dataBuffer[20];
  //sprintf(indexBuffer,"W[%08u] = ",position);
  //sprintf(dataBuffer,"0x%08x",*pointer);
  //ret = string(indexBuffer)+string(dataBuffer);

  ret = index(position) + getHexString(*pointer);
  
  return ret;	
}


/*-------------------------------------------------*/
/* DCCDataParser::~DCCDataParser                   */
/* destructor                                      */
/*-------------------------------------------------*/
DCCDataParser::~DCCDataParser(){
  
  // delete DCCEvents if any...
  vector<DCCEventBlock *>::iterator it;
  for(it=dccEvents_.begin();it!=dccEvents_.end();it++){delete *it;}
  dccEvents_.clear();
    
  delete mapper_;
}
