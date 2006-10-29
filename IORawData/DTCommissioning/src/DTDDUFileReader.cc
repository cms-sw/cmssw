/** \file
 *
 *  $Date: 2006/08/03 14:46:06 $
 *  $Revision: 1.8 $
 *  \author M. Zanetti
 */

#include <IORawData/DTCommissioning/src/DTDDUFileReader.h>
#include <IORawData/DTCommissioning/src/DTFileReaderHelpers.h>

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

#include <DataFormats/Common/interface/EventID.h>
#include <DataFormats/Common/interface/Timestamp.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <string>
#include <iosfwd>
#include <iostream>
#include <algorithm>
   
using namespace std;
using namespace edm;


DTDDUFileReader::DTDDUFileReader(const edm::ParameterSet& pset) : 
  runNumber(1), eventNumber(0) {

  const string & filename = pset.getParameter<string>("fileName");

  readFromDMA = pset.getUntrackedParameter<bool>("isRaw",false);

  inputFile.open(filename.c_str());
  if( inputFile.fail() ) {
    throw cms::Exception("InputFileMissing") 
      << "[DTDDUFileReader]: the input file: " << filename <<" is not present";
  } else {
    cout << "DTDDUFileReader: DaqSource file '" << filename << "' was succesfully opened" << endl;
  }
  
  //else if (readFromDMA) 
  inputFile.ignore(4*7);
  
  skipEvents = pset.getUntrackedParameter<int>("skipEvents",0);
  if (skipEvents) { 
    cout<<""<<endl;
    cout<<"   Dear user, pleas be patient, "<<skipEvents<<" are being skipped .."<<endl;
    cout<<""<<endl;
  }

}


DTDDUFileReader::~DTDDUFileReader(){
      inputFile.close();
}

bool DTDDUFileReader::fillRawData(EventID& eID,
				  Timestamp& tstamp, 
				  FEDRawDataCollection*& data){

  vector<uint64_t> eventData;
  size_t estimatedEventDimension = 1024; // dimensione hardcoded
  eventData.reserve(estimatedEventDimension); 
  uint64_t word = 0;

  bool haederTag = false;
  bool dataTag = true;
  
  
  int wordCount = 0;
  
  // getting the data word by word from the file
  // do it until you get the DDU trailer
  while ( !isTrailer(word, dataTag, wordCount) ) {
    //while ( !isTrailer(word) ) { 
    
    if (readFromDMA) {
      int nread;
      word = dmaUnpack(dataTag,nread);
      if ( nread<=0 ) {
	cout<<"[DTDDUFileReader]: ERROR! failed to get the trailer"<<endl;
	return false;
      }
    }
    
    else {
      int nread = inputFile.read(dataPointer<uint64_t>( &word ), dduWordLength);
      dataTag = false;
      if ( nread<=0 ) {
	cout<<"[DTDDUFileReader]: ERROR! failed to get the trailer"<<endl;
	return false;
      }
    }
    
    // get the DDU header
    if (isHeader(word,dataTag)) haederTag=true;
    
    // from now on fill the eventData with the ROS data
    if (haederTag) {
      
      if (readFromDMA) {
	// swapping only the 8 byte words
	if (dataTag) {
	  swap(word);
	} // WARNING also the ddu status words have been swapped!
	// Control the correct interpretation in DDUUnpacker
      }
      
      eventData.push_back(word);
      wordCount++;
    }
    
  } 
  
  //     FEDTrailer candidate(reinterpret_cast<const unsigned char*>(&word));
  //     cout<<"EventSize: pushed back "<<eventData.size()
  // 	<<";  From trailer "<<candidate.lenght()<<endl;
  
  // next event reading will start with meaningless trailer+header from DTLocalDAQ
  // those will be skipped automatically when seeking for the DDU header
  //if (eventData.size() > estimatedEventDimension) throw 2;
  
  // Eventually skipping events
  if ((int)eventNumber >= skipEvents) {

    // Setting the Event ID
    eID = EventID( runNumber, eventNumber);
    
    // eventDataSize = (Number Of Words)* (Word Size)
    int eventDataSize = eventData.size()*dduWordLength;
    
    // The FED ID is always the first in the DT range
    FEDRawData& fedRawData = data->FEDData( FEDNumbering::getDTFEDIds().first );
    fedRawData.resize(eventDataSize);
    
    copy(reinterpret_cast<unsigned char*>(&eventData[0]),
	 reinterpret_cast<unsigned char*>(&eventData[0]) + eventDataSize, fedRawData.data());

  }

  return true;
    
}

void DTDDUFileReader::swap(uint64_t & word) {
  
  twoNibble64* newWorld = reinterpret_cast<twoNibble64*>(&word);

  uint32_t msBits_tmp = newWorld->msBits;
  newWorld->msBits = newWorld->lsBits;
  newWorld->lsBits = msBits_tmp;
}


uint64_t DTDDUFileReader::dmaUnpack(bool & isData, int & nread) {
  
  uint64_t dduWord = 0;

  uint32_t td[4];
  // read 4 32-bits word from the file;
  nread = inputFile.read(dataPointer<uint32_t>( &td[0] ), 4);
  nread += inputFile.read(dataPointer<uint32_t>( &td[1] ), 4);
  nread += inputFile.read(dataPointer<uint32_t>( &td[2] ), 4);
  nread += inputFile.read(dataPointer<uint32_t>( &td[3] ), 4);

  uint32_t data[2];
  // adjust 4 32-bits words  into 2 32-bits words
  data[0] |= td[3] & 0x3ffff;
  data[0] |= (td[2] << 18 ) & 0xfffc0000;
  data[1] |= (td[2] >> 14 ) & 0x0f;
  data[1] |= (td[1] << 4 ) & 0x3ffff0;
  data[1] |= (td[0] << 22 ) & 0xffc00000;

  isData = ( td[0] >> 10 ) & 0x01;

  // push_back to a 64 word
  dduWord = (uint64_t(data[1]) << 32) | data[0];

  return dduWord;
}

bool DTDDUFileReader::isHeader(uint64_t word, bool dataTag) {

  bool it_is = false;
  FEDHeader candidate(reinterpret_cast<const unsigned char*>(&word));
  if ( candidate.check() ) {
    // if ( candidate.check() && !dataTag) {
    it_is = true;
   eventNumber++;
  }
 
  return it_is;
}


bool DTDDUFileReader::isTrailer(uint64_t word, bool dataTag, int wordCount) {

  bool it_is = false;
  FEDTrailer candidate(reinterpret_cast<const unsigned char*>(&word));
  if ( candidate.check() ) {
    //  if ( candidate.check() && !dataTag) {
    //cout<<"[DTDDUFileReader] "<<wordCount<<" - "<<candidate.lenght()<<endl;
    if ( wordCount == candidate.lenght())
      it_is = true;
  }
 
  return it_is;
}


bool DTDDUFileReader::checkEndOfFile(){

  bool retval=false;
  if ( inputFile.eof() ) retval=true;
  return retval;

}



