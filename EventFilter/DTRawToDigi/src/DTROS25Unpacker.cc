/** \file
 *
 *  $Date: 2006/01/20 15:44:34 $
 *  $Revision: 1.6 $
 *  \author  M. Zanetti - INFN Padova 
 */

#include <EventFilter/DTRawToDigi/src/DTROS25Unpacker.h>
#include <EventFilter/DTRawToDigi/src/DTDDUWords.h>
#include <EventFilter/DTRawToDigi/src/DTROSErrorNotifier.h>
#include <EventFilter/DTRawToDigi/src/DTTDCErrorNotifier.h>
#include <CondFormats/DTObjects/interface/DTReadOutMapping.h>

#include <iostream>

using namespace std;
using namespace edm;


void DTROS25Unpacker::interpretRawData(const unsigned int* index, int datasize,
				       int dduID,
				       edm::ESHandle<DTReadOutMapping>& mapping, 
				       std::auto_ptr<DTDigiCollection>& product) {


  const int wordLength = 4;
  int numberOfWords = datasize / wordLength;

  int rosID = 1; // To be taken from DDU control word

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
      int eventCounter = rosHeaderWord.TTCEventCounter();
      cout<<"eventCounter "<<eventCounter<<endl;
      
      rosID++; // to be mapped;
	  
      // Loop on ROBs
      do {	  
	wordCounter++; word = index[wordCounter];

 	// Eventual ROS Error: occurs when some errors are found in a ROB
 	if (DTROSWordType(word).type() == DTROSWordType::ROSError) {
 	  DTROSErrorWord dtROSErrorWord(word);
 	  DTROSErrorNotifier dtROSError(dtROSErrorWord);
 	  dtROSError.print();
 	} 

	// Eventual ROS Debugging; 
	else if (DTROSWordType(word).type() == DTROSWordType::ROSDebug) {
	  DTROSDebugWord rosDebugWord(word);
	  cout<<"ROS Debug type "<<rosDebugWord.debugType() <<endl;
	  cout<<"ROS Debug message "<<rosDebugWord.debugMessage() <<endl;
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
	    cout<<"ROB Event Id (trailer) "<<robTrailerWord.eventID()<<endl;
	    cout<<"ROB WordCount (trailer) "<<robTrailerWord.wordCount()<<endl;
	    cout<<"ROB ID (trailer) "<<robTrailerWord.robID()<<endl;
	    
	  }
 	}

      } while ( DTROSWordType(word).type() != DTROSWordType::ROSTrailer );

      // check ROS Trailer (condition already verified)
      if (DTROSWordType(word).type() == DTROSWordType::ROSTrailer){
	DTROSTrailerWord rosTrailerWord(word);
	cout<<"ROS Trailer TFF "<<rosTrailerWord.TFF()<<endl;
	cout<<"ROS Trailer TXP "<<rosTrailerWord.TPX()<<endl;
	cout<<"ROS Trailer ECHO "<<rosTrailerWord.ECHO()<<endl;
	cout<<"ROS Trailer ECLO "<<rosTrailerWord.ECLO()<<endl;
	cout<<"ROS Trailer BCO "<<rosTrailerWord.BCO()<<endl;
	cout<<"ROS Trailer Event Counter "<<rosTrailerWord.EventWordCount()<<endl;
      }
    }

    // (needed only if there are more than 1 ROS
    wordCounter++; word = index[wordCounter];
  }  
  
}
