/** \file
 * Implementation of class RPCUnpackingModule
 *
 *  $Date: 2006/02/06 09:25:19 $
 *  $Revision: 1.10 $
 *
 * \author Ilaria Segoni
 */

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

#include <EventFilter/RPCRawToDigi/interface/RPCMonitorInterface.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>

using namespace edm;
using namespace std;

RPCUnpackingModule::RPCUnpackingModule(const edm::ParameterSet& pset) 
:nEvents(0){
  
  printout = pset.getUntrackedParameter<bool>("PrintOut", false); 
  hexprintout = pset.getUntrackedParameter<bool>("PrintHexDump", false); 
  
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

 if(printout) cout<<"Entering RPCUnpackingModule::produce"<<endl;
 
 /// Get Data from all FEDs
 Handle<FEDRawDataCollection> allFEDRawData; 
 e.getByLabel("DaqRawData", allFEDRawData); 
 if(printout) cout<<"Got FEDRawData"<<endl;
 
 auto_ptr<RPCDigiCollection> producedRPCDigis(new RPCDigiCollection);

 std::pair<int,int> rpcFEDS=FEDNumbering::getRPCFEDIds();
 if(printout) cout<<"Starting loop on FEDs, RPC FED ID RANGE: "<<rpcFEDS.first<<" - "<<rpcFEDS.second<<endl;
 
 nEvents++; 
 
 if(printout) cout<<"Beginning To Unpack Event: "<<nEvents<<endl;
 
	RPCRecordFormatter interpreter(printout);
	for (int id= rpcFEDS.first; id<=rpcFEDS.second; ++id){  

 		const FEDRawData & fedData = allFEDRawData->FEDData(id);
   
 		if(printout) cout<<"Beginning to Unpack FED number "<<id<<", FED size: "<<fedData.size()<<" bytes"<<endl;			 

		if(fedData.size()){
     
     			const unsigned char* index = fedData.data();
       
      			/// Unpack FED Header(s)
      			int numberOfHeaders= this->unpackHeader(index);
 
      			/// Unpack FED Trailer(s)
      			const unsigned char* trailerIndex=index+fedData.size()- rpc::unpacking::SLINK_WORD_SIZE;
      			int numberOfTrailers=this->unpackTrailer(trailerIndex);
       
      			if(printout) cout<<"Found "<<numberOfHeaders<<" Headers and "<<numberOfTrailers<<" Trailers"<<endl;		  
      
      			/// Beginning of RPC Records Unpacking
      			index += numberOfHeaders* rpc::unpacking::SLINK_WORD_SIZE; 
       
      			/// Loop on S-LINK words        
      			 while( index != trailerIndex ){            
         
	 			/// Loop on RPC Records
         			int numOfRecords=0;
	 			//RPCRecord::recordTypes expectedRecord=RPCRecord::UndefinedType;
         
	 			for(int nRecord=0; nRecord<4; nRecord++){
          
	 				 const unsigned char* recordIndex;
	  				 recordIndex=index+ rpc::unpacking::SLINK_WORD_SIZE -(nRecord+1)* rpc::unpacking::RPC_RECORD_SIZE;
	   
	   				if(hexprintout) {	  
	     					numOfRecords++;
	     					const unsigned int* word;	  
             					word=reinterpret_cast<const unsigned int*>(recordIndex);
	     					cout<<oct<<*word<<" ";
	     					if(numOfRecords==4) {
	        					cout<<endl;
	      					}		
	   			 	}	
          
	  				RPCRecord theRecord(recordIndex,printout);
        
          				/// Find out type of record
          				RPCRecord::recordTypes typeOfRecord = theRecord.type();
	 
	  				/// Unpack the Record 	  
	  				interpreter.recordUnpack(typeOfRecord,recordIndex,producedRPCDigis);	  
	 		 	}
          
          			///Go to beginning of next word
          			index+= rpc::unpacking::SLINK_WORD_SIZE;

      			}
		
      			///Send information to DQM
      			if(instatiateDQM){ 
         			 RPCEventData rpcEvent= interpreter.eventData();
          			 monitor->process(rpcEvent);
      			}	
		}
		
	}       
        // Insert the new product in the event  
	e.put(producedRPCDigis);  
     
}





int RPCUnpackingModule::unpackHeader(const unsigned char* headerIndex) const {
 
 int numberOfHeaders=0;
 bool moreHeaders=true;
 
 while(moreHeaders){
  FEDHeader fedHeader(headerIndex); 
  numberOfHeaders++;
  
  if(printout) cout<<"Trigger type: "<<fedHeader.triggerType()
     <<", L1A: "<<fedHeader.lvl1ID()
     <<", BX: "<<fedHeader.bxID()
     <<", soucre ID: "<<fedHeader.sourceID()
     <<", version: "<<fedHeader.version()
     <<", more headers: "<<fedHeader.moreHeaders()
     <<endl;
  
  moreHeaders=fedHeader.moreHeaders();
  headerIndex++;
 
 }
  
 return numberOfHeaders;    		

}




int RPCUnpackingModule::unpackTrailer(const unsigned char* trailerIndex) const {
 
 int numberOfTrailers=0;
 bool moreTrailers = true;
 
 while(moreTrailers){
 
  FEDTrailer fedTrailer(trailerIndex);
  
  if(fedTrailer.check())  {
       
       numberOfTrailers++;	   
       trailerIndex -= rpc::unpacking::SLINK_WORD_SIZE;     

       if(printout) cout<<"Trailer length: "<< fedTrailer.lenght()<<
          " CRC "<<fedTrailer.crc()<<
          " Event Fragment Status "<< fedTrailer.evtStatus()<<
          " Value of Trigger Throttling System "<<fedTrailer.ttsBits()<<
          " more Trailers: "<<fedTrailer.moreTrailers()<<endl;
    
  
       
  }else{
       moreTrailers=false;
  }

 }
 
  
 return numberOfTrailers;
}




DEFINE_FWK_MODULE(RPCUnpackingModule)
