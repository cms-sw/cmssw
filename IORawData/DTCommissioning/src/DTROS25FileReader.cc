/** \file
 *
 *  $Date: 2010/03/12 10:04:19 $
 *  $Revision: 1.14 $
 *  \author M. Zanetti
 */

#include <IORawData/DTCommissioning/src/DTROS25FileReader.h>
#include <IORawData/DTCommissioning/src/DTFileReaderHelpers.h>

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

#include "DataFormats/Provenance/interface/EventID.h" 	 
#include <DataFormats/Provenance/interface/Timestamp.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <string>
#include <iosfwd>
#include <iostream>
#include <algorithm>
   
using namespace std;
using namespace edm;


DTROS25FileReader::DTROS25FileReader(const edm::ParameterSet& pset) : 
  runNumber(1), eventNumber(0) {
      
  const string & filename = pset.getUntrackedParameter<string>("fileName");

  inputFile.open(filename.c_str());
  if( inputFile.fail() ) {
    throw cms::Exception("InputFileMissing") 
      << "DTROS25FileReader: the input file: " << filename <<" is not present";
  }
}


DTROS25FileReader::~DTROS25FileReader(){
  inputFile.close();
}


int DTROS25FileReader::fillRawData(EventID& eID,
				   Timestamp& tstamp, 
				   FEDRawDataCollection*& data){
  data = new FEDRawDataCollection();

  vector<uint32_t> eventData;
  size_t estimatedEventDimension = 102400; // dimensione hardcoded
  eventData.reserve(estimatedEventDimension); 
  uint32_t word = 0;
  

  try {   

    bool marked = false;

    // getting the data word by word from the file
    // do it until you get the ROS25 trailer
    while ( !isTrailer(word) ) { 
      
      // get the first word
      int nread = inputFile.read(dataPointer<uint32_t>( &word ), rosWordLenght);
      
      // WARNING!!! ||swapping it|| (Check whether it is necessary) 
      swap(word);

      if ( nread<=0 ) throw 1;

      // get the ROS25 header
      if (isHeader(word)) marked=true;

      // from now on fill the eventData with the ROS data
      if (marked) {
	eventData.push_back(word);

      }
    } 

    // next event reading will start with meaningless trailer+header from DTLocalDAQ
    // those will be skipped automatically when seeking for the ROS25 header

    //if (eventData.size() > estimatedEventDimension) throw 2;
    
    // Setting the Event ID
    eID = EventID( runNumber, 1U, eventNumber);

    // eventDataSize = (Number Of Words)* (Word Size)
    int eventDataSize = eventData.size()*rosWordLenght;
    // It has to be a multiple of 8 bytes. if not, adjust the size of the FED payload
    int adjustment = (eventDataSize/4)%2 == 1 ? 4 : 0; 

    // The FED ID is always the first in the DT range
    FEDRawData& fedRawData = data->FEDData( FEDNumbering::MINDTFEDID );
    fedRawData.resize(eventDataSize+adjustment);
    
    copy(reinterpret_cast<unsigned char*>(&eventData[0]),
	 reinterpret_cast<unsigned char*>(&eventData[0]) + eventDataSize, fedRawData.data());

    return true;
  }

  catch( int i ) {

    if ( i == 1 ){
      cout<<"[DTROS25FileReader]: ERROR! failed to get the trailer"<<endl;
      delete data; data=0;
      return false;
    }    
    else {
      cout<<"[DTROS25FileReader]:"
	  <<" ERROR! ROS data exceeding estimated event dimension. Event size = "
	  <<eventData.size()<<endl;
      delete data; data=0;
      return false;
    }
    
  }

}

void DTROS25FileReader::swap(uint32_t & word) {
  
  twoNibble* newWorld = reinterpret_cast<twoNibble*>(&word);

  uint16_t msBits_tmp = newWorld->msBits;
  newWorld->msBits = newWorld->lsBits;
  newWorld->lsBits = msBits_tmp;
}


bool DTROS25FileReader::isHeader(uint32_t word) {

  bool it_is = false;
  if ( (word >> 24 ) == 31 ) {
    it_is = true;
    ++eventNumber;
  }
 
  return it_is;
}


bool DTROS25FileReader::isTrailer(uint32_t word) {
 
  bool it_is = false;
  if ( (word >> 24 ) == 63 ) {
    it_is = true;
  }
 
  return it_is;
}


bool DTROS25FileReader::checkEndOfFile(){

  bool retval=false;
  if ( inputFile.eof() ) retval=true;
  return retval;

}



