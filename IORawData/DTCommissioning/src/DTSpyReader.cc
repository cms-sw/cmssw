/** \file
 *
 *  $Date: 2007/06/25 08:07:52 $
 *  $Revision: 1.12 $
 *  \author M. Zanetti
 */

#include <IORawData/DTCommissioning/src/DTSpyReader.h>
#include <IORawData/DTCommissioning/src/DTFileReaderHelpers.h>

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

#include "DataFormats/Provenance/interface/EventID.h"
#include <DataFormats/Provenance/interface/Timestamp.h>

#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <string>
#include <iosfwd>
#include <iostream>
#include <algorithm>
#include <stdio.h>

   
using namespace std;
using namespace edm;


DTSpyReader::DTSpyReader(const edm::ParameterSet& pset) : 
  runNumber(1), eventNumber(0) {

  string connectionParameters = pset.getParameter<string>("fileName");

  mySpy = new DTSpy(); 

  /// connecting to XDAQ note ("0.0.0.0" = localhost)
  mySpy->Connect(connectionParameters.c_str(),10000);  

}


DTSpyReader::~DTSpyReader() {
  delete mySpy;
}


bool DTSpyReader::fillRawData(EventID& eID, Timestamp& tstamp, FEDRawDataCollection*& data){
  
  // get the pointer to the beginning of the buffer
  char * rawDTData = mySpy->getEventPointer();
  
  uint32_t * rawDTData32 = reinterpret_cast<uint32_t*>(rawDTData);

  // instantiate the FEDRawDataCollection
  data = new FEDRawDataCollection();

  FEDHeader * dduHeader;

  vector<uint64_t> eventData;  uint64_t word = 0;  int wordCount = 0;

  bool haederTag = false; bool dataTag = true;
  
  // getting the data word by word from the file
  // do it until you get the DDU trailer
  while ( !isTrailer(word, dataTag, wordCount) ) {
    
    word = dmaUnpack(rawDTData32, dataTag);
    
    // get the DDU header
    if (isHeader(word,dataTag)) {
      haederTag=true;
      dduHeader = new FEDHeader(reinterpret_cast<const unsigned char*>(&word));      
    }

    // from now on fill the eventData with the ROS data
    if (haederTag) {
      
      // swapping only the 32 bits words
      if (dataTag) swap(word);
      // WARNING also the ddu status words have been swapped!
      // Control the correct interpretation in DDUUnpacker
      
      eventData.push_back(word);
      wordCount++;
    }
  }
 
  // Setting the Event ID
  eID = EventID( runNumber, eventNumber);
  
  // eventDataSize = (Number Of Words)* (Word Size)
  int eventDataSize = eventData.size()*dduWordLength;
  
  // Get the FED ID from the DDU 
  FEDRawData& fedRawData = data->FEDData( dduHeader->sourceID() );
  fedRawData.resize(eventDataSize);
  
  copy(reinterpret_cast<unsigned char*>(&eventData[0]),
       reinterpret_cast<unsigned char*>(&eventData[0]) + eventDataSize, fedRawData.data());
  
  return true;
    
}

void DTSpyReader::swap(uint64_t & word) {
  
  twoNibble64* newWorld = reinterpret_cast<twoNibble64*>(&word);

  uint32_t msBits_tmp = newWorld->msBits;
  newWorld->msBits = newWorld->lsBits;
  newWorld->lsBits = msBits_tmp;
}


uint64_t DTSpyReader::dmaUnpack(uint32_t* dmaData, bool & isData) {
  
  uint64_t dduWord = 0;

  uint32_t unpackedData[2];
  // adjust 4 32-bits words  into 2 32-bits words
  unpackedData[0] |= dmaData[3] & 0x3ffff;
  unpackedData[0] |= (dmaData[2] << 18 ) & 0xfffc0000;
  unpackedData[1] |= (dmaData[2] >> 14 ) & 0x0f;
  unpackedData[1] |= (dmaData[1] << 4 ) & 0x3ffff0;
  unpackedData[1] |= (dmaData[0] << 22 ) & 0xffc00000;

  isData = ( dmaData[0] >> 10 ) & 0x01;

  // push_back to a 64 word
  dduWord = (uint64_t(unpackedData[1]) << 32) | unpackedData[0];

  return dduWord;
}

bool DTSpyReader::isHeader(uint64_t word, bool dataTag) {

  bool it_is = false;
  FEDHeader candidate(reinterpret_cast<const unsigned char*>(&word));
  if ( candidate.check() ) {
    // if ( candidate.check() && !dataTag) {
    it_is = true;
   eventNumber++;
  }
 
  return it_is;
}


bool DTSpyReader::isTrailer(uint64_t word, bool dataTag, int wordCount) {

  bool it_is = false;
  FEDTrailer candidate(reinterpret_cast<const unsigned char*>(&word));
  if ( candidate.check() ) {
    //  if ( candidate.check() && !dataTag) {
    //cout<<"[DTSpyReader] "<<wordCount<<" - "<<candidate.lenght()<<endl;
    if ( wordCount == candidate.lenght())
      it_is = true;
  }
 
  return it_is;
}






