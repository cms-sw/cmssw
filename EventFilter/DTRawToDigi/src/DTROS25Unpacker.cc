/** \file
 *
 *  $Date: 2005/11/21 17:38:48 $
 *  $Revision: 1.2 $
 *  \author  M. Zanetti - INFN Padova 
 */

#include <EventFilter/DTRawToDigi/src/DTROS25Unpacker.h>

#include <EventFilter/DTRawToDigi/src/DTDDUWords.h>
#include <EventFilter/DTRawToDigi/src/DTROSErrorNotifier.h>
#include <EventFilter/DTRawToDigi/src/DTTDCErrorNotifier.h>
#include <CondFormats/DTObjects/interface/DTReadOutMapping.h>

using namespace std;
using namespace edm;

#include <iostream>

#define SLINK_WORD_SIZE 8


void DTROS25Unpacker::interpretRawData(const unsigned char* index, int datasize,
				       int dduID,
				       edm::ESHandle<DTReadOutMapping>& mapping, 
				       std::auto_ptr<DTDigiCollection>& product) {

  // Set the index to start looping on ROS data
  index += SLINK_WORD_SIZE - DTDDU_WORD_SIZE;
  DTROSWordType wordType(index);	

  // Loop on ROSs
  int rosID = 0;
  do {
    index+=DTDDU_WORD_SIZE;
    wordType.update();

    // ROS Header; 
    if (wordType.type() == DTROSWordType::ROSHeader) {
      DTROSHeaderWord rosHeaderWord(index);
      int eventCounter = rosHeaderWord.TTCEventCounter();

      rosID++; // to be mapped;
	  
      // Loop on ROBs
      do {	  
	index+=DTDDU_WORD_SIZE;
	wordType.update();
	    
	// Eventual ROS Error: occurs when some errors are found in a ROB
	if (wordType.type() == DTROSWordType::ROSError) {
	  DTROSErrorWord dtROSErrorWord(index);
	  DTROSErrorNotifier dtROSError(dtROSErrorWord);
	  dtROSError.print();
	} 
	    
	// Check ROB header	  
	else if (wordType.type() == DTROSWordType::GroupHeader) {
	       
	  DTROBHeaderWord robHeaderWord(index);
	  int eventID = robHeaderWord.eventID(); // from the TDCs
	  int bunchID = robHeaderWord.bunchID(); // from the TDCs
	  int robID = robHeaderWord.robID(); // to be mapped
	      
	  // Loop on TDCs data (headers and trailers are not there
	  do {
	    index+=DTDDU_WORD_SIZE;
	    wordType.update();
		
	    // Eventual TDC Error 
	    if ( wordType.type() == DTROSWordType::TDCError) {
	      DTTDCErrorWord dtTDCErrorWord(index);
	      DTTDCErrorNotifier dtTDCError(dtTDCErrorWord);
	      dtTDCError.print();
	    } 
		
	    // The TDC information
	    else if (wordType.type() == DTROSWordType::TDCMeasurement) {
	      DTTDCMeasurementWord tdcMeasurementWord(index);
		  
	      int tdcID = tdcMeasurementWord.tdcID(); 
	      int tdcChannel = tdcMeasurementWord.tdcChannel(); 
		  
	      // Map the RO channel to the DetId and wire
	      DTDetId layer; int wire = 0;
	      //mapping->getId(dduID, rosID, robID, tdcID, tdcChannel, layer, wire);
		  
	      // Produce the digi
	      DTDigi digi( tdcMeasurementWord.tdcTime(), wire);
	      product->insertDigi(layer,digi);
	    }
		
	  } while ( wordType.type() != DTROSWordType::GroupTrailer );

	  // Check ROB Trailer (condition already verified)
	  if (wordType.type() == DTROSWordType::GroupTrailer) ;
	}

      } while ( wordType.type() != DTROSWordType::ROSTrailer );

      // check ROS Trailer (condition already verified)
      if (wordType.type() == DTROSWordType::ROSTrailer);
    }

  } while (index != (index + datasize - 3*SLINK_WORD_SIZE));


}
