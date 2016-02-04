/** \file
 *
 *  $Date: 2010/02/03 16:58:24 $
 *  $Revision: 1.10 $
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

  // instatiating Sandro's spy (My name is Bond, Sandro Bond)
  mySpy = new  DTSpy(); 

  /// connecting to XDAQ note ("0.0.0.0" = localhost)
  string connectionParameters = pset.getUntrackedParameter<string>("connectionParameters");
  mySpy->Connect(connectionParameters.c_str(),10000);  

  cout<<endl;
  cout<<"DT Local DAQ online spy. Connected to IP "<<connectionParameters.c_str()
      <<". Waiting for the data to be flushed"<<endl;
  cout<<endl;

  debug = pset.getUntrackedParameter<bool>("debug",false);
  dduID = pset.getUntrackedParameter<int32_t>("dduID",770); // NOT needed
}


DTSpyReader::~DTSpyReader() {
  delete mySpy;
}


int DTSpyReader::fillRawData(EventID& eID, Timestamp& tstamp, FEDRawDataCollection*& data){
  
  // ask for a new buffer
  mySpy->getNextBuffer();

  // get the pointer to the beginning of the buffer
  const char * rawDTData = mySpy->getEventPointer();
  
  const uint32_t * rawDTData32 = reinterpret_cast<const uint32_t*>(rawDTData);

  // instantiate the FEDRawDataCollection
  data = new FEDRawDataCollection();

  vector<uint64_t> eventData;  uint64_t word = 0;  
  int wordCount = 0; int wordCountCheck = 0;

  bool headerTag = false; bool dataTag = true;
  
  // Advance at long-word steps until the trailer is reached. 
  // Skipped whatever else is in the buffer (e.g. another event..)
  while ( !isTrailer(word, dataTag, wordCount) ) {
    
    // dma gets 4 32-bits words and create a 64 bit one
    word = dmaUnpack(rawDTData32, dataTag);

    // look for the DDU header
    if (isHeader(word,dataTag))  headerTag=true;

    // check whether the first word is a DDU header
    if ( wordCountCheck > 0 && !headerTag && debug) 
      cout<<"[DTSpyReader]: WARNING: header still not found!!"<<endl;

    // from now on fill the eventData with the ROS data
    if (headerTag) {
      
      // swapping only the 32 bits words
      if (dataTag) swap(word);
      // WARNING also the ddu status words have been swapped!
      // Control the correct interpretation in DDUUnpacker
      
      eventData.push_back(word);
      wordCount++; 
    }

    // advancing by 4 32-bits words
    rawDTData32 += 4;

    // counting the total number of group of 128 bits (=4*32) in the buffer 
    wordCountCheck++; 
  }

  // Setting the Event ID
  runNumber = mySpy->getRunNo(); 
  eID = EventID( runNumber, 1U, eventNumber);
  
  // eventDataSize = (Number Of Words)* (Word Size)
  int eventDataSize = eventData.size()*dduWordLength; 
  
  if (debug) cout<<" DDU ID = "<<dduID<<endl;
 
  FEDRawData& fedRawData = data->FEDData( dduID );
  fedRawData.resize(eventDataSize);
  
  copy(reinterpret_cast<unsigned char*>(&eventData[0]),
       reinterpret_cast<unsigned char*>(&eventData[0]) + eventDataSize, fedRawData.data());


  mySpy->setlastPointer( (char*) rawDTData32 );

  return true;
    
}

void DTSpyReader::swap(uint64_t & word) {
  
  twoNibble64* newWorld = reinterpret_cast<twoNibble64*>(&word);

  uint32_t msBits_tmp = newWorld->msBits;
  newWorld->msBits = newWorld->lsBits;
  newWorld->lsBits = msBits_tmp;
}


uint64_t DTSpyReader::dmaUnpack(const uint32_t* dmaData, bool & isData) {
  
  uint32_t unpackedData[2] = {0, 0};
  // adjust 4 32-bits words  into 2 32-bits words
  unpackedData[0] |= dmaData[3] & 0x3ffff;
  unpackedData[0] |= (dmaData[2] << 18 ) & 0xfffc0000;
  unpackedData[1] |= (dmaData[2] >> 14 ) & 0x0f;
  unpackedData[1] |= (dmaData[1] << 4 ) & 0x3ffff0;
  unpackedData[1] |= (dmaData[0] << 22 ) & 0xffc00000;

  isData = ( dmaData[0] >> 10 ) & 0x01;

  //printf ("Lower part: %x \n", unpackedData[0]);
  //printf ("Upper part: %x \n", unpackedData[1]);

  // push_back to a 64 word
  uint64_t dduWord = ( uint64_t(unpackedData[1]) << 32 ) | unpackedData[0];

  return dduWord;
}

bool DTSpyReader::isHeader(uint64_t word, bool dataTag) {

  bool it_is = false;
  FEDHeader candidate(reinterpret_cast<const unsigned char*>(&word));
  if ( candidate.check() ) {
    // if ( candidate.check() && !dataTag) {
    it_is = true;
    dduID = candidate.sourceID();
    eventNumber++;
  }
  return it_is;
}


bool DTSpyReader::isTrailer(uint64_t word, bool dataTag, int wordCount) {

  bool it_is = false;
  FEDTrailer candidate(reinterpret_cast<const unsigned char*>(&word));
  if ( candidate.check() ) {
    //  if ( candidate.check() && !dataTag) {
    if ( wordCount == candidate.lenght())
      it_is = true;
  }
  return it_is;
}






