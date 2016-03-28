/** \file
 *
 *  $Date: 2015/07/10 
 *  \author M.C Fouz 
 *  Updated from class DTROS8FileReader to a new class: DTNewROS8FileReader  
 *    to read ROS8 for MB1 GIF++ 2015 data  (for CMSSW_5X versions)
 *    include also the PU Data (Chamber Trigger info)
 *  
 */


#include <IORawData/DTCommissioning/plugins/DTNewROS8FileReader.h>
#include <IORawData/DTCommissioning/plugins/DTFileReaderHelpers.h>

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


DTNewROS8FileReader::DTNewROS8FileReader(const edm::ParameterSet& pset) : 
  runNumber(1), eventNum(0) {
      
  const string & filename = pset.getUntrackedParameter<string>("fileName");

  inputFile.open(filename.c_str());
  if( inputFile.fail() ) {
    throw cms::Exception("InputFileMissing") 
      << "DTNewROS8FileReader: the input file: " << filename <<" is not present";
  }
 
  produces<FEDRawDataCollection>();
}


DTNewROS8FileReader::~DTNewROS8FileReader(){
  inputFile.close();
}


int DTNewROS8FileReader::fillRawData(Event& e,
//				  Timestamp& tstamp, 
				  FEDRawDataCollection*& data){
  EventID eID = e.id();
  data = new FEDRawDataCollection();

  try {
    /* Structure of the DATA 

     1.- NUMBER OF WORDS (it includes this word and the last counter)
     2.- HEADER: 8 Words
           Header word 0: run number
           Header word 1: spill number
           Header word 2: event number
           Header word 3: reserved
           Header word 4: ROS data offset
           Header word 5: PU data offset
           Header word 6: reserved
           Header word 7: reserved
     3.- ROS DATA ==> DECODE BY THE DTROS8Unpacker.cc in EventFilter 
           3.1 NUMBER OF ROS WORDS (it includes this counter)
           3.2 Lock status Word
           3.3 ROS DATA
                 3.3.1 ROS BOARD ID/ROS CHANNEL - 1 word
                 3.3.2 GLOBAL HEADER - 1 word
                 3.3.3 TDC Data Words - X words (depends on event)
                 3.3.4 GLOBAL TRAILER
     4.- PU DATA (trigger) ==> DECODE BY THE DTROS8Unpacker.cc in EventFilter  
                 Not always. If chamber is autotriggered doesn't have PU data except
                             the Number of PU words
           4.1.- NUMBER OF WORDS (it includes this word, always present 
                                  even if there is not PU data)
           4.2.- PATTERN UNIT ID - 1 word (data counter & PU-ID)
           4.3.- DATA - X Words ?????
     5.- NUMBER OF WORDS (include this word and the first counter)
    */


    if( checkEndOfFile() ) throw 1; 
    
    // Number of words in the header, including the first word of header that is the number of words in the event
    int numberEventHeadWords=8;

    //1.- Get the total NUMBER OF WORDs from the 1st word in the payload
    int numberOfWords=0;
    int nread = 0;
    nread = inputFile.read(dataPointer<int>( &numberOfWords ), ros8WordLenght); // ros8WordLength=4
    if ( nread<=0 ) throw 1;

    // inputFile.ignore(4*(numberEventHeadWords-1)); // Skip the header. The first word of header has been already read 
       
    //2.- Get the HEADER  ============================================================================
    int datahead[numberEventHeadWords];
    for(int iih=0;iih<numberEventHeadWords;iih++)
    {
        nread = inputFile.read(dataPointer<int>( &datahead[iih] ), ros8WordLenght); 
    }

    //3.- ROS DATA  &  4.- PU DATA (Trigger)   =======================================================
    // Get the event data (all words but the header and the first and the last counter)
    int numberOfDataWords=numberOfWords-numberEventHeadWords-2; 
    int* eventData = new int[numberOfDataWords];
    nread = inputFile.read(dataPointer<int>( eventData ), numberOfDataWords * ros8WordLenght );
    if ( nread<=0 ) throw 1;

    //5.- Get the total NUMBER OF WORDs from the last word in the payload
    // Check that the event data size corresponds to the 1st word datum 
    int LastCounter=0;
    nread=inputFile.read(dataPointer<int>( &LastCounter ), ros8WordLenght); 
    if ( nread<=0 ) throw 1;
      
    if ( LastCounter != numberOfWords ) {
      cout << "[DTNewROS8FileReader]: word counter mismatch exception: "
	   << numberOfWords << " " << LastCounter << endl;
      throw 99;
    }

   //The first word in the header is the run number 
   runNumber= datahead[0];
      cout << "[DTNewROS8FileReader]: Run Number: "<<dec<<runNumber<<endl;
     
   //The third word in the header is the event number (without any reset)
   //eventNum= datahead[2];  //francos system
   eventNum= datahead[1];  //linux system
   //cout<<"ëventNum  "<<dec<<eventNum<<endl;
   if(eventNum<1)eventNum=1;// Event number must start at 1 but at TDCs it starts at cero,if not the program crashes
                           // files used for testing start in 1, but... just in case...

   eID = EventID(runNumber, 1U, eventNum);  

   //cout << " EEEEE eID: " << eID << endl;
   // Even if we have introducing the correct runNumber when running the runNumber appears always as =1

   int eventDataSize = numberOfDataWords * ros8WordLenght;
   int adjustment = (eventDataSize/4)%2 == 1 ? 4 : 0;   

   // The FED ID is always the first in the DT range
   FEDRawData& fedRawData = data->FEDData( FEDNumbering::MINDTFEDID );
   fedRawData.resize(eventDataSize+adjustment);  // the size must be multiple of 8 bytes
    
   // Passing the data to the Event
   copy(reinterpret_cast<unsigned char*>(eventData), 
	 reinterpret_cast<unsigned char*>(eventData) + eventDataSize, fedRawData.data());

   // needed to get rid of memory leaks (?)
   delete[] eventData;

   return true;

  }
  catch( int i ) {

    if ( i == 1 ){
      cout << "[DTNewROS8FileReader]: END OF FILE REACHED. "
           << "No information read for the requested event" << endl;
      delete data; data=0;
      return false;
    }    
    else {
      cout << "[DTNewROS8FileReader]: PROBLEM WITH EVENT INFORMATION ON THE FILE. "
           << "EVENT DATA READING FAILED  code= " << i << endl;
      delete data; data=0;
      return false;
    }

  }

}


void DTNewROS8FileReader::produce(Event&e, EventSetup const&es){

   edm::Handle<FEDRawDataCollection> rawdata;
   FEDRawDataCollection *fedcoll = 0;
   fillRawData(e,fedcoll);
   std::auto_ptr<FEDRawDataCollection> bare_product(fedcoll);
   e.put(bare_product);
}


bool DTNewROS8FileReader::checkEndOfFile(){

  bool retval=false;
  if ( inputFile.eof() ) retval=true;
  return retval;

}

