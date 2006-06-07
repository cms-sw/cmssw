/** \file
 * Implementation of class RPCUnpackingModule
 *
 *  $Date: 2006/03/30 15:19:30 $
 *  $Revision: 1.15 $
 *
 * \author Ilaria Segoni
 */


//#define DEBUG_RPCUNPACKER

#include <EventFilter/RPCRawToDigi/interface/RPCUnpackingModule.h>
#include <EventFilter/RPCRawToDigi/interface/RPCUnpackingParameters.h>
#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>
#include <EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h>

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h> 
#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/RPCRawToDigi/interface/RPCMonitorInterface.h"

#include <iostream>

using namespace edm;

RPCUnpackingModule::RPCUnpackingModule(const edm::ParameterSet& pset) 
:nEvents(0){
  
  printout = pset.getUntrackedParameter<bool>("PrintOut", false); 
  
  instatiateDQM = pset.getUntrackedParameter<bool>("runDQM", false);
  if(instatiateDQM){   
     monitor = edm::Service<RPCMonitorInterface>().operator->();   
  }
  
  produces<RPCDigiCollection>();

}

RPCUnpackingModule::~RPCUnpackingModule(){
  delete monitor;
  monitor = 0;
}


void RPCUnpackingModule::produce(Event & e, const EventSetup& c){

 edm::LogInfo ("RPCUnpacker") <<"Entering RPCUnpackingModule::produce";
 
 /// Get Data from all FEDs
 Handle<FEDRawDataCollection> allFEDRawData; 
 e.getByType(allFEDRawData); 

 edm::LogInfo ("RPCUnpacker") <<"Got FEDRawData";
 
 std::auto_ptr<RPCDigiCollection> producedRPCDigis(new RPCDigiCollection);

 std::pair<int,int> rpcFEDS=FEDNumbering::getRPCFEDIds();
 edm::LogInfo ("RPCUnpacker") <<"Starting loop on FEDs, RPC FED ID RANGE: "<<rpcFEDS.first<<" - "<<rpcFEDS.second;
 
 nEvents++; 
 
 edm::LogInfo ("RPCUnpacker") <<"Beginning To Unpack Event: "<<nEvents;

	RPCRecordFormatter interpreter;
	for (int id= rpcFEDS.first; id<=rpcFEDS.second; ++id){  

 		const FEDRawData & fedData = allFEDRawData->FEDData(id);
    		RPCFEDData rpcRawData;
  
                edm::LogInfo ("RPCUnpacker") <<"Beginning to Unpack FED number "<<id<<", FED size: "<<fedData.size()<<" bytes";			 

		if(fedData.size()){
     
     			const unsigned char* index = fedData.data();
       
      			/// Unpack FED Header(s)
      			int numberOfHeaders= this->unpackHeader(index, rpcRawData);
 
      			/// Unpack FED Trailer(s)
      			const unsigned char* trailerIndex=index+fedData.size()- rpc::unpacking::SLINK_WORD_SIZE;
      			int numberOfTrailers=this->unpackTrailer(trailerIndex, rpcRawData);
       
      			edm::LogInfo ("RPCUnpacker") <<"Found "<<numberOfHeaders<<" Headers and "<<numberOfTrailers<<" Trailers";		  
      
      			/// Beginning of RPC Records Unpacking
      			index += numberOfHeaders* rpc::unpacking::SLINK_WORD_SIZE; 
       
      			/// Loop on S-LINK words        
      			 while( index != trailerIndex ){            
         
	 			/// Loop on RPC Records
         			int numOfRecords=0;
	 			enum RPCRecord::recordTypes previousRecord=RPCRecord::UndefinedType;

	 			for(int nRecord=0; nRecord<4; nRecord++){
          
	 				 const unsigned char* recordIndex;
	  				 recordIndex=index+ rpc::unpacking::SLINK_WORD_SIZE -(nRecord+1)* rpc::unpacking::RPC_RECORD_SIZE;
	   
	   				        #ifdef DEBUG_RPCUNPACKER	  
	     					numOfRecords++;
	     					const unsigned int* word;	  
             					word=reinterpret_cast<const unsigned int*>(recordIndex);
	     					cout<<oct<<*word<<" ";
	     					if(numOfRecords==4) {
	        					cout<<std::endl;
	      					}		
	   			 	        #endif	
          
	  				RPCRecord theRecord(recordIndex,printout,previousRecord);
        
          				/// Find out type of record
          				RPCRecord::recordTypes typeOfRecord = theRecord.computeType();					
					/// Check Record is of expected type
	 				bool missingRecord = theRecord.check();
	  				/// Unpack the Record 	  
	  				interpreter.recordUnpack(theRecord,producedRPCDigis,rpcRawData);	  
	 		 	}
          
          			///Go to beginning of next word
          			index+= rpc::unpacking::SLINK_WORD_SIZE;

      			}
		
      			///Send information to DQM
      			if(instatiateDQM){ 
          			 monitor->process(rpcRawData);
      			}	
		}
		
	}       
        // Insert the new product in the event  
	e.put(producedRPCDigis);  
     
}





int RPCUnpackingModule::unpackHeader(const unsigned char* headerIndex, RPCFEDData & rawData) {
 
 int numberOfHeaders=0;
 bool moreHeaders=true;
 
 while(moreHeaders){
  FEDHeader fedHeader(headerIndex); 
  rawData.addCdfHeader(fedHeader);
  numberOfHeaders++;
  edm::LogInfo ("RPCUnpacker") <<"Trigger type: "<<fedHeader.triggerType()
     <<", L1A: "<<fedHeader.lvl1ID()
     <<", BX: "<<fedHeader.bxID()
     <<", soucre ID: "<<fedHeader.sourceID()
     <<", version: "<<fedHeader.version()
     <<", more headers: "<<fedHeader.moreHeaders();
  
  moreHeaders=fedHeader.moreHeaders();
  headerIndex++;
 
 }
  
 return numberOfHeaders;    		

}




int RPCUnpackingModule::unpackTrailer(const unsigned char* trailerIndex, RPCFEDData & rawData) {
 
 int numberOfTrailers=0;
 bool moreTrailers = true;
 
 while(moreTrailers){
 
  FEDTrailer fedTrailer(trailerIndex);
  rawData.addCdfTrailer(fedTrailer);
  
  if(fedTrailer.check())  {
       
       numberOfTrailers++;	   
       trailerIndex -= rpc::unpacking::SLINK_WORD_SIZE;     

       edm::LogInfo ("RPCUnpacker") <<"Trailer length: "<< fedTrailer.lenght()<<
          " CRC "<<fedTrailer.crc()<<
          " Event Fragment Status "<< fedTrailer.evtStatus()<<
          " Value of Trigger Throttling System "<<fedTrailer.ttsBits()<<
          " more Trailers: "<<fedTrailer.moreTrailers();
    
  
       
  }else{
       moreTrailers=false;
  }

 }
 
  
 return numberOfTrailers;
}




DEFINE_FWK_MODULE(RPCUnpackingModule)
