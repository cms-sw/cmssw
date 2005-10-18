/** \file
 *
 *  $Date: 2005/10/07 09:01:46 $
 *  $Revision: 1.4 $
 *  \author S. Argiro - N. Amapane - M. Zanetti 
 */


#include <EventFilter/DTRawToDigi/src/DTUnpackingModule.h>
#include <EventFilter/DTRawToDigi/src/DTDaqCMSFormatter.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>

#include <EventFilter/DTRawToDigi/src/DTDDUWords.h>

using namespace edm;
using namespace std;

#include <iostream>


#define SLINK_WORD_SIZE 8


DTUnpackingModule::DTUnpackingModule(const edm::ParameterSet& pset) : 
  formatter(new DTDaqCMSFormatter()) {
  produces<DTDigiCollection>();
}

DTUnpackingModule::~DTUnpackingModule(){
 delete formatter;
}


void DTUnpackingModule::produce(Event & e, const EventSetup& c){

  Handle<FEDRawDataCollection> rawdata;
  e.getByLabel("DaqRawData", rawdata);

  // create the collection of MB Digis
  auto_ptr<DTDigiCollection> product(new DTDigiCollection);

  
  for (int id=FEDNumbering::getDTFEDIds().first; id<=FEDNumbering::getDTFEDIds().second; ++id){ 

    const FEDRawData& feddata = rawdata->FEDData(id);

    if (feddata.size()){

      const unsigned char* index = feddata.data();
      
      // Interpret FED header and trailer, check consistency, etc.
      // header  : index
//       FEDHeader fedheader(index);
//       // look into it

//       // DDU status 1: index+feddata.size() - 3*SLINK_WORD_SIZE      
//       // DDU status 2: index+feddata.size() - 2*SLINK_WORD_SIZE
//       // FED trailer : 
//       FEDTrailer(index+feddata.size()-SLINK_WORD_SIZE);
//       //look into it

      index += SLINK_WORD_SIZE - DTDDU_WORD_SIZE;
      DTROSWordType wordType(index);	
      
      // Loop on ROSs
      do {

	index+=DTDDU_WORD_SIZE;
	wordType.update();

	// Check ROS Header; 
	if (wordType.type() == DTROSWordType::ROSHeader) {
	  DTROSHeaderWord rosHeaderWord(index);
	  int eventCounter = rosHeaderWord.TTCEventCounter();
 	  // Check it with the DDU Eventcounter. Else???
	}
	
 	// Loop on ROBs
 	do {	  
 	  index+=DTDDU_WORD_SIZE;
 	  wordType.update();
 	  // Check ROB header	  
 	  if (wordType.type() == DTROSWordType::GroupHeader) {
	    
 	    DTROBHeaderWord robHeaderWord(index);
 	    int robID = robHeaderWord.robID(); // to be mapped
 	    int eventID = robHeaderWord.eventID(); // to be checked with the previuos ones
 	    int bunchID = robHeaderWord.bunchID(); // to be checked with the DDU one

 	    // Loop on TDCs
  	    do {
  	      index+=DTDDU_WORD_SIZE;
  	      wordType.update();

  	      // Check TDC header	  	    
  	      if (wordType.type() == DTROSWordType::TDCHeader) {
  		DTTDCHeaderWord tdcHeaderWord(index);

  		// some information as ROB header but for:
  		int tdcID = tdcHeaderWord.tdcID(); // to be mapped

  		do {
  		  index+=DTDDU_WORD_SIZE;
  		  wordType.update();
  		  // Check the TDC Measurement
  		  if (wordType.type() == DTROSWordType::TDCMeasurement) {
		    DTTDCMeasurementWord tdcMeasurementWord(index);
		    int tdcTime = tdcMeasurementWord.tdcTime(); // THE DATUM
  		  }
  		} while ( wordType.type() != DTROSWordType::TDCTrailer );

  		// Check the TDC Trailer
  		index+=DTDDU_WORD_SIZE;
  		wordType.update();
  		if (wordType.type() == DTROSWordType::TDCTrailer) ;
  	      }
  	      else if ( wordType.type() == DTROSWordType::TDCError) {
  		DTTDCErrorWord tdcErrorWord(index);
  		cout<<"[DTUnpackingModule]: WARNING!! TDC Error of type "<<tdcErrorWord.tdcError()
  		    <<", from TDC "<<tdcErrorWord.tdcID()<<endl;
  	      } 

  	    } while ( wordType.type() != DTROSWordType::GroupTrailer );
 	    // Check ROB Trailer
 	    index+=DTDDU_WORD_SIZE;
 	    if (wordType.type() == DTROSWordType::GroupTrailer) ;
 	  }

 	  else if (wordType.type() == DTROSWordType::ROSError) {
 	    DTROSErrorWord rosErrorWord(index);
 	    cout<<"[DTUnpackingModule]: WARNING!! ROS Error of type "<<rosErrorWord.errorType()
 		<<", from ROB "<<rosErrorWord.robID()<<endl;
 	  } 

	} while ( wordType.type() != DTROSWordType::ROSTrailer );
 	// check ROS Trailer      
 	if (wordType.type() == DTROSWordType::ROSTrailer);

      } while (index != (feddata.data()+feddata.size()-2*SLINK_WORD_SIZE));
	
	
      
    }
  } 
  // commit to the event  
  //  e.put(product);
}

