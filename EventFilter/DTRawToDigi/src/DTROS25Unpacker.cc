/** \file
 *
 *  $Date: 2006/02/14 17:08:16 $
 *  $Revision: 1.7 $
 *  \author  M. Zanetti - INFN Padova 
 */

#include <EventFilter/DTRawToDigi/src/DTROS25Unpacker.h>

#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>
#include <EventFilter/DTRawToDigi/interface/DTROS25Data.h>

#include <EventFilter/DTRawToDigi/interface/DTDataMonitorInterface.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <EventFilter/DTRawToDigi/src/DTROSErrorNotifier.h>
#include <EventFilter/DTRawToDigi/src/DTTDCErrorNotifier.h>

// Mapping
#include <CondFormats/DTObjects/interface/DTReadOutMapping.h>

#include <iostream>

using namespace std;
using namespace edm;



DTROS25Unpacker::DTROS25Unpacker(const edm::ParameterSet& ps): pset(ps) {

  if(pset.getUntrackedParameter<bool>("performDataIntegrityMonitor",false)){
    dataMonitor = edm::Service<DTDataMonitorInterface>().operator->(); 
  }

}


void DTROS25Unpacker::interpretRawData(const unsigned int* index, int datasize,
				       int dduID,
				       edm::ESHandle<DTReadOutMapping>& mapping, 
				       std::auto_ptr<DTDigiCollection>& product) {


  const int wordLength = 4;
  int numberOfWords = datasize / wordLength;

  int rosID = 0; // To be taken from DDU control word
  DTROS25Data controlData(rosID);

  int wordCounter = 0;
  uint32_t word = index[wordCounter];


  /******************************************************
  / The the loop is performed with "do-while" statements
  / because the ORDER of the words in the event data
  / is assumed to be fixed. Eventual changes into the 
  / structure should be considered as data corruption
  *******************************************************/

  // Loop on ROSs
  while (wordCounter < numberOfWords) {

    cout<<"Word Type "<<DTROSWordType(word).type()<<endl;

    // ROS Header; 
    if (DTROSWordType(word).type() == DTROSWordType::ROSHeader) {
      DTROSHeaderWord rosHeaderWord(word);

      rosID++; // to be mapped;

      // container for words to be sent to DQM
      controlData.setROSId(rosID);

      // Loop on ROBs
      do {	  
	wordCounter++; word = index[wordCounter];

 	// Eventual ROS Error: occurs when some errors are found in a ROB
 	if (DTROSWordType(word).type() == DTROSWordType::ROSError) {
 	  DTROSErrorWord dtROSErrorWord(word);
	  controlData.addROSError(dtROSErrorWord);
 	} 

	// Eventual ROS Debugging; 
	else if (DTROSWordType(word).type() == DTROSWordType::ROSDebug) {
	  DTROSDebugWord rosDebugWord(word);
	  controlData.addROSDebug(rosDebugWord);
	}

	// Check ROB header	  
 	else if (DTROSWordType(word).type() == DTROSWordType::GroupHeader) {
	  
 	  DTROBHeaderWord robHeaderWord(word);
 	  int eventID = robHeaderWord.eventID(); // from the TDCs
	  cout<<"ROB Event Id "<<eventID<<endl;

 	  int bunchID = robHeaderWord.bunchID(); // from the TDCs
	  cout<<"ROB bunch ID "<<bunchID<<endl;

 	  int robID = robHeaderWord.robID(); // to be mapped
	  cout<<"ROB ID "<<robID<<endl;

 	  // Loop on TDCs data (headers and trailers are not there)
 	  do {
	    wordCounter++; word = index[wordCounter];
		
 	    // Eventual TDC Error 
 	    if ( DTROSWordType(word).type() == DTROSWordType::TDCError) {
 	      DTTDCErrorWord dtTDCErrorWord(word);
 	      DTTDCErrorNotifier dtTDCError(dtTDCErrorWord);
 	      dtTDCError.print();
 	    }  		
 	    // Eventual TDC Debug
 	    else if ( DTROSWordType(word).type() == DTROSWordType::TDCDebug) {
	      cout<<"TDC Debugging"<<endl;
 	    }  		
 	    // The TDC information
 	    else if (DTROSWordType(word).type() == DTROSWordType::TDCMeasurement) {
 	      DTTDCMeasurementWord tdcMeasurementWord(word);
		  
 	      int tdcID = tdcMeasurementWord.tdcID(); 
	      cout<<"TDC ID "<<tdcID<<endl;

 	      int tdcChannel = tdcMeasurementWord.tdcChannel(); 
	      cout<<"TDC Channel "<<tdcChannel<<endl;

	      cout<<"TDC Time "<<tdcMeasurementWord.tdcTime()<<endl;

 	      // Map the RO channel to the DetId and wire
 	      DTLayerId layer; int wire = 0;
 	      // mapping->getId(dduID, rosID, robID, tdcID, tdcChannel, layer, wire);
		  
 	      // Produce the digi
	      // DTDigi digi( tdcMeasurementWord.tdcTime(), wire);
	      // product->insertDigi(layer,digi);
 	    }
		
 	  } while ( DTROSWordType(word).type() != DTROSWordType::GroupTrailer );
	  
 	  // Check ROB Trailer (condition already verified)
 	  if (DTROSWordType(word).type() == DTROSWordType::GroupTrailer) {

	    DTROBTrailerWord robTrailerWord(word);
	  }
 	}

      } while ( DTROSWordType(word).type() != DTROSWordType::ROSTrailer );

      // check ROS Trailer (condition already verified)
      if (DTROSWordType(word).type() == DTROSWordType::ROSTrailer){
	DTROSTrailerWord rosTrailerWord(word);
	
	controlData.addROSTrailer(rosTrailerWord);

      }
    }

    // (needed only if there are more than 1 ROS
    wordCounter++; word = index[wordCounter];
  }  
  
  // Perform dqm if requested
  if (pset.getUntrackedParameter<bool>("performDataIntegrityMonitor",false)) {
    dataMonitor->process(controlData);
  } 

}
