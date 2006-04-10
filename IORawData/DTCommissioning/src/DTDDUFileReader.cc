/** \file
 *
 *  $Date: 2006/03/28 08:59:50 $
 *  $Revision: 1.3 $
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

  inputFile.open(filename.c_str(), ios::in | ios::binary );
  if( inputFile.fail() ) {
    throw cms::Exception("InputFileMissing") 
      << "DTDDUFileReader: the input file: " << filename <<" is not present";
  }
}


DTDDUFileReader::~DTDDUFileReader(){
  inputFile.close();
}


bool DTDDUFileReader::fillRawData(EventID& eID,
				  Timestamp& tstamp, 
				  FEDRawDataCollection& data){

  vector<uint64_t> eventData;
  size_t estimatedEventDimension = 1024; // dimensione hardcoded
  eventData.reserve(estimatedEventDimension); 
  uint64_t word = 0;
  
  try {   

    bool marked = false;

    // getting the data word by word from the file
    // do it until you get the DDU trailer
    while ( !isTrailer(word) ) { 

      // get the first word
      inputFile.read(dataPointer<uint64_t>( &word ), dduWordLenght);

      if ( !inputFile ) throw 1;

      // get the DDU header
      if (isHeader(word)) marked=true;

      // from now on fill the eventData with the ROS data
      if (marked) {
	eventData.push_back(word);

      }
    } 

    // next event reading will start with meaningless trailer+header from DTLocalDAQ
    // those will be skipped automatically when seeking for the DDU header

    if (eventData.size() > estimatedEventDimension) throw 2;
    
    // Setting the Event ID
    eID = EventID( runNumber, eventNumber);

    // eventDataSize = (Number Of Words)* (Word Size)
    int eventDataSize = eventData.size()*dduWordLenght;
    // It has to be a multiple of 8 bytes. if not, adjust the size of the FED payload
    int adjustment = (eventDataSize/4)%2 == 1 ? 4 : 0; 

    // The FED ID is always the first in the DT range
    FEDRawData& fedRawData = data.FEDData( FEDNumbering::getDTFEDIds().first );
    fedRawData.resize(eventDataSize+adjustment);
    
    copy(reinterpret_cast<unsigned char*>(&eventData[0]),
	 reinterpret_cast<unsigned char*>(&eventData[0]) + eventDataSize, fedRawData.data());

    return true;
  }

  catch( int i ) {

    if ( i == 1 ){
      cout<<"[DTDDUFileReader]: ERROR! failed to get the trailer"<<endl;
      return false;
    }    
    else {
      cout<<"[DTDDUFileReader]:"
	  <<" ERROR! ROS data exceeding estimated event dimension. Event size = "
	  <<eventData.size()<<endl;
      return false;
    }
    
  }

}

void DTDDUFileReader::swap(uint64_t & word) {
  
  twoNibble64* newWorld = reinterpret_cast<twoNibble64*>(&word);

  uint32_t msBits_tmp = newWorld->msBits;
  newWorld->msBits = newWorld->lsBits;
  newWorld->lsBits = msBits_tmp;
}


bool DTDDUFileReader::isHeader(uint64_t word) {

  bool it_is = false;
  FEDHeader candidate(reinterpret_cast<const unsigned char*>(&word));
  if ( candidate.check()) {
    it_is = true;
    eventNumber++;
  }
 
  return it_is;
}


bool DTDDUFileReader::isTrailer(uint64_t word) {

  bool it_is = false;
  FEDTrailer candidate(reinterpret_cast<const unsigned char*>(&word));
  if ( candidate.check()) {
    it_is = true;
  }
 
  return it_is;
}


bool DTDDUFileReader::checkEndOfFile(){

  bool retval=false;
  if ( inputFile.eof() ) retval=true;
  return retval;

}



