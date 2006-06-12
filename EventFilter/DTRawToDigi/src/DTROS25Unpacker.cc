/** \file
 *
 *  $Date: 2006/06/12 10:27:50 $
 *  $Revision: 1.16 $
 *  \author  M. Zanetti - INFN Padova 
 */

#include <EventFilter/DTRawToDigi/src/DTROS25Unpacker.h>

#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>
#include <EventFilter/DTRawToDigi/interface/DTControlData.h>
#include <EventFilter/DTRawToDigi/interface/DTROChainCoding.h>

#include <EventFilter/DTRawToDigi/interface/DTDataMonitorInterface.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <EventFilter/DTRawToDigi/src/DTROSErrorNotifier.h>
#include <EventFilter/DTRawToDigi/src/DTTDCErrorNotifier.h>

// Mapping
#include <CondFormats/DTObjects/interface/DTReadOutMapping.h>

#include <iostream>
#include <math.h>

using namespace std;
using namespace edm;


// FIXME: SC words processing is missing!!


DTROS25Unpacker::DTROS25Unpacker(const edm::ParameterSet& ps): pset(ps) {

  if(pset.getUntrackedParameter<bool>("performDataIntegrityMonitor",false)){
    cout<<"[DTROS25Unpacker]: Enabling Data Integrity Checks"<<endl;
    dataMonitor = edm::Service<DTDataMonitorInterface>().operator->(); 
  }

  debug = pset.getUntrackedParameter<bool>("debugMode",false);

}


DTROS25Unpacker::~DTROS25Unpacker() {
  cout<<"[DTROS25Unpacker]: Destructor"<<endl;
}

void DTROS25Unpacker::interpretRawData(const unsigned int* index, int datasize,
                                       int dduID,
                                       edm::ESHandle<DTReadOutMapping>& mapping, 
                                       std::auto_ptr<DTDigiCollection>& product,
				       uint16_t rosList) {


  /// FIXME! (temporary). The DDU number is set by hand
  dduID = pset.getUntrackedParameter<int>("dduID",730);

  const int wordLength = 4;
  int numberOfWords = datasize / wordLength;

  int rosID = 0; 
  DTROS25Data controlData(rosID);

  int wordCounter = 0;
  uint32_t word = index[wordCounter];

  map<uint32_t,int> hitOrder;


  /******************************************************
  / The the loop is performed with "do-while" statements
  / because the ORDER of the words in the event data
  / is assumed to be fixed. Eventual changes into the 
  / structure should be considered as data corruption
  *******************************************************/

  // Loop on ROSs
  while (wordCounter < numberOfWords) {

    rosID++; // to be mapped;
    
    if ( pset.getUntrackedParameter<bool>("readingDDU",true) ) {
      // matching the ROS number with the enabled DDU channel
      if ( rosID <= 12 && !((rosList & int(pow(2., (rosID-1) )) ) >> (rosID-1) ) ) continue;      
      
      if (debug) cout<<"[DTROS25Unpacker]: ros list: "<<rosList
		     <<" ROS ID "<<rosID<<endl;
    }
    
    // ROS Header; 
    if (DTROSWordType(word).type() == DTROSWordType::ROSHeader) {
      DTROSHeaderWord rosHeaderWord(word);

      if (debug) cout<<"[DTROS25Unpacker]: ROSHeader "<<rosHeaderWord.TTCEventCounter()<<endl;

      // container for words to be sent to DQM
      controlData.setROSId(rosID);

      // Loop on ROBs
      do {        
        wordCounter++; word = index[wordCounter];

        // Eventual ROS Error: occurs when some errors are found in a ROB
        if (DTROSWordType(word).type() == DTROSWordType::ROSError) {
          DTROSErrorWord dtROSErrorWord(word);
          controlData.addROSError(dtROSErrorWord);
	  if (debug) cout<<"[DTROS25Unpacker]: ROSError, Error type "<<dtROSErrorWord.errorType()
			 <<" robID "<<dtROSErrorWord.robID()<<endl;
        } 

        // Eventual ROS Debugging; 
        else if (DTROSWordType(word).type() == DTROSWordType::ROSDebug) {
          DTROSDebugWord rosDebugWord(word);
          controlData.addROSDebug(rosDebugWord);
	  if (debug) cout<<"[DTROS25Unpacker]: ROSDebug, type "<<rosDebugWord.debugType()
			 <<"  message "<<rosDebugWord.debugMessage()<<endl;
        }

        // Check ROB header       
        else if (DTROSWordType(word).type() == DTROSWordType::GroupHeader) {
          
          DTROBHeaderWord robHeaderWord(word);
	  int eventID = robHeaderWord.eventID(); // from the TDCs
	  int bunchID = robHeaderWord.bunchID(); // from the TDCs
          int robID = robHeaderWord.robID(); // to be mapped

	  if (debug) cout<<"[DTROS25Unpacker] ROB: ID "<<robID
			 <<" Event ID "<<eventID
			 <<" BXID "<<bunchID<<endl;

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
              controlData.addTDCMeasurement(tdcMeasurementWord);
              DTTDCData tdcData(robID,tdcMeasurementWord);
              controlData.addTDCData(tdcData);
              
	      int tdcID = tdcMeasurementWord.tdcID(); 
              int tdcChannel = tdcMeasurementWord.tdcChannel(); 

	      if (debug) cout<<"[DTROS25Unpacker] TDC: ID "<<tdcID
			     <<" Channel "<< tdcChannel
			     <<" Time "<<tdcMeasurementWord.tdcTime()<<endl;

	      DTROChainCoding channelIndex(dduID, rosID, robID, tdcID, tdcChannel);

	      hitOrder[channelIndex.getCode()]++;


	      if (debug) {
		cout<<"[DTROS25Unpacker] ROAddress: DDU "<< dduID 
		    <<", ROS "<< rosID
		    <<", ROB "<< robID
		    <<", TDC "<< tdcID
		    <<", Channel "<< tdcChannel<<endl;
	      }
	    
              // Map the RO channel to the DetId and wire
 	      DTWireId detId = mapping->readOutToGeometry(dduID, rosID, robID, tdcID, tdcChannel);
 	      if (debug) cout<<"[DTROS25Unpacker] "<<detId<<endl;
 	      int wire = detId.wire();

	      // Produce the digi
	      DTDigi digi( wire, tdcMeasurementWord.tdcTime(), hitOrder[channelIndex.getCode()]-1);
              product->insertDigi(detId.layerId(),digi);
            }
                
          } while ( DTROSWordType(word).type() != DTROSWordType::GroupTrailer );
          
          // Check ROB Trailer (condition already verified)
          if (DTROSWordType(word).type() == DTROSWordType::GroupTrailer) {
            DTROBTrailerWord robTrailerWord(word);
            controlData.addROBTrailer(robTrailerWord);
	    if (debug) cout<<"[DTROS25Unpacker]: ROBTrailer, robID  "<<robTrailerWord.robID()
			   <<" eventID  "<<robTrailerWord.eventID()
			   <<" wordCount  "<<robTrailerWord.wordCount()<<endl;
          }
        }

	// Check the eventual Sector Collector Header       
        else if (DTROSWordType(word).type() == DTROSWordType::SCHeader) {
	  DTLocalTriggerHeaderWord scHeaderWord(word);
	  if (debug) cout<<"[DTROS25Unpacker]: SCHeader  eventID "<<scHeaderWord.eventID()<<endl;

	  int bx_counter=0;

	  do {
	    bx_counter++;
            wordCounter++; word = index[wordCounter];
  	    if (DTROSWordType(word).type() == DTROSWordType::SCData) {
	      DTLocalTriggerDataWord scDataWord(word);
	      if (debug) {
		//cout<<"[DTROS25Unpacker]: SCData bits "<<scDataWord.SCData()<<endl;
		if (scDataWord.hasTrigger(0)) 
		  cout<<" at BX "<<round(bx_counter/2.)
		      <<" lower part has trigger! with track quality "<<scDataWord.trackQuality(0)<<endl;
		if (scDataWord.hasTrigger(1)) 
		  cout<<" at BX "<<round(bx_counter/2.)
		      <<" upper part has trigger! with track quality "<<scDataWord.trackQuality(1)<<endl;
	      }
	    }

	  } while ( DTROSWordType(word).type() != DTROSWordType::SCTrailer );

	  if (DTROSWordType(word).type() == DTROSWordType::SCTrailer) {
	    DTLocalTriggerTrailerWord scTrailerWord(word);
	    if (debug) cout<<"[DTROS25Unpacker]: SCTrailer, number of words "<<scTrailerWord.wordCount()<<endl;
	  }
	}

      } while ( DTROSWordType(word).type() != DTROSWordType::ROSTrailer );

      // check ROS Trailer (condition already verified)
      if (DTROSWordType(word).type() == DTROSWordType::ROSTrailer){
        DTROSTrailerWord rosTrailerWord(word);
        controlData.addROSTrailer(rosTrailerWord);
	if (debug) cout<<"[DTROS25Unpacker]: ROSTrailer "<<rosTrailerWord.EventWordCount()<<endl;
      }

      // Perform dqm if requested:
      // DQM IS PERFORMED FOR EACH ROS SEPARATELY
      if (pset.getUntrackedParameter<bool>("performDataIntegrityMonitor",false)) {
	dataMonitor->processROS25(controlData, dduID, rosID);
      }

    }

    else if (index[wordCounter] == 0) {
      // in the case of odd number of words of a given ROS the header of 
      // the next one is postponed by 4 bytes word set to 0.
      // rosID needs to be step back by 1 unit
      if (debug) cout<<"[DTROS25Unpacker]: odd number of ROS words"<<endl;
      rosID--;
    }

    else {
      cout<<"[DTROS25Unpacker]: ERROR! First word is not a ROS Header"<<endl;
    }


    // (needed if there are more than 1 ROS)
    wordCounter++; word = index[wordCounter];

  }  
  
  
}
