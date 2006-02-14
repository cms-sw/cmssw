/** \file
 *
 *  $Date: 2005/11/25 18:14:12 $
 *  $Revision: 1.4 $
 *  \author M. Zanetti
 */

#include <IORawData/DTCommissioning/src/DTROS8FileReader.h>
#include <IORawData/DTCommissioning/src/DTFileReaderHelpers.h>

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

#include <FWCore/EDProduct/interface/EventID.h>
#include <FWCore/EDProduct/interface/Timestamp.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <string>
#include <iosfwd>
#include <iostream>
#include <algorithm>
   
using namespace std;
using namespace edm;


DTROS8FileReader::DTROS8FileReader(const edm::ParameterSet& pset) : 
  runNum(1), eventNum(0) {
      
  const string & filename = pset.getParameter<string>("fileName");

  inputFile.open(filename.c_str(), ios::in | ios::binary );
  if( inputFile.fail() ) {
    throw cms::Exception("InputFileMissing") 
      << "DTROS8FileReader: the input file: " << filename <<" is not present";
  }
}


DTROS8FileReader::~DTROS8FileReader(){
  inputFile.close();
}


bool DTROS8FileReader::fillRawData(EventID& eID,
				   Timestamp& tstamp, 
				   FEDRawDataCollection& data){


  try {
    

    if( checkEndOfFile() ) throw 1; 
    

    // Get the total number of words from the 1st word in the payload
    int numberOfWords;
    inputFile.read(dataPointer<int>( &numberOfWords ), ros8WordLenght);
    if ( !inputFile )  throw 1;


    // Get the event data (all words but the 1st)
    int* eventData = new int[numberOfWords];
    inputFile.read(dataPointer<int>( eventData + 1 ), (numberOfWords-1) * ros8WordLenght );
    if ( !inputFile ) throw 1;
    

    // Check that the event data size corresponds to the 1st word datum 
    if ( eventData[numberOfWords-1] != numberOfWords ) {
      cout << "[DTROS8FileReader]: word counter mismatch exception: "
	   << numberOfWords << " " << eventData[numberOfWords-1] << endl;
      throw 99;
    }

    // The header added by the local DAQ occupies 8 words, starting from the 2nd 
    int* head = eventData + 1;
    
    /* 
      Header word 0: run number
      Header word 1: spill number
      Header word 2: event number
      Header word 3: reserved
      Header word 4: ROS data offset
      Header word 5: PU data offset
      Header word 6: reserved
      Header word 7: reserved
    */

    // WARNING: the event number is reset at a new spill
    eID = EventID( head[0], head[1]*head[2]);

    // The pointer to the ROS payload (the 1st word being the ROS words counter)
    int* rosData = eventData + head[4];

    // The ROS payload size
    int eventDataSize = *rosData * ros8WordLenght;


    // The FED ID is always the first in the DT range
    FEDRawData& fedRawData = data.FEDData( FEDNumbering::getDTFEDIds().first );
    fedRawData.resize(eventDataSize);
    
    // I pass only the ROS data to the Event
    copy(reinterpret_cast<unsigned char*>(rosData), 
	 reinterpret_cast<unsigned char*>(rosData) + eventDataSize, fedRawData.data());

    // needed to get rid of memory leaks (?)
    delete[] eventData;

    return true;
  }

  catch( int i ) {

    if ( i == 1 ){
      cout << "[DTROS8FileReader]: END OF FILE REACHED. "
           << "No information read for the requested event" << endl;
      return false;
    }    
    else {
      cout << "[DTROS8FileReader]: PROBLEM WITH EVENT INFORMATION ON THE FILE. "
           << "EVENT DATA READING FAILED  code= " << i << endl;
      return false;
    }

  }

}


bool DTROS8FileReader::checkEndOfFile(){

  bool retval=false;
  if ( inputFile.eof() ) retval=true;
  return retval;

}



