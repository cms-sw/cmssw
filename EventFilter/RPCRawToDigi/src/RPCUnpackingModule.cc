/* 
 *
 *  
 *  
 */

#include <EventFilter/RPCRawToDigi/interface/RPCUnpackingModule.h>
#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>
#include <EventFilter/RPCRawToDigi/interface/StartOfBXData.h>
#include <EventFilter/RPCRawToDigi/interface/StartOfChannelData.h>
#include <EventFilter/RPCRawToDigi/interface/ChamberData.h>
#include <EventFilter/RPCRawToDigi/interface/RMBErrorData.h>

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>

using namespace edm;
using namespace std;


#include <iostream>

#define SLINK_WORD_SIZE 8



RPCUnpackingModule::RPCUnpackingModule(const edm::ParameterSet& pset)  
 {
 
  produces<RPCDigiCollection>();

 }

RPCUnpackingModule::~RPCUnpackingModule(){
}


void RPCUnpackingModule::produce(Event & e, const EventSetup& c){

  Handle<FEDRawDataCollection> allFEDRawData; 
  e.getByLabel("DaqRawData", allFEDRawData); //FED Raw data for all feds in the event


  auto_ptr<RPCDigiCollection> producedRPCDigis(new RPCDigiCollection);


    
	const std::pair<int,int> rpcFEDS=FEDNumbering::getMRpcFEDIds();
	for (int id= rpcFEDS.first; id<=rpcFEDS.second; ++id){ 

		const FEDRawData & fedData = allFEDRawData->FEDData(id);
		
		
		if(fedData.size()){
		        
			int numberOfHeaders=1;
			
      			const unsigned char* index = fedData.data();
			//FEDHeader fedHeader(index); 
			
			const unsigned char* trailerIndex=index+fedData.size();
		 	//FEDTrailer fedTrailer(trailerIndex);
			

// Beginning of RPC Records Unpacking
			index += numberOfHeaders*SLINK_WORD_SIZE; 


			while( index != trailerIndex ){
			
			 RPCRecord theRecord(index);
			 
			 // Find what type of record it is
			 RPCRecord::recordTypes typeOfRecord = theRecord.type();
			 
			 int bx=0;
			 if(typeOfRecord==RPCRecord::StartOfBXData)
                         {
			  StartOfBXData bxData(index);
			 } 
			 if(typeOfRecord==RPCRecord::StartOfChannelData)
			 {
			    StartOfChannelData channelData(index);
			 }


			 if(typeOfRecord==RPCRecord::ChamberData)
			 {
			    ChamberData chambData(index);
			    //int rpcChamber = chambData.chamberNumber();
			    //int partitionNumber = chambData.partitionNumber();
			    //int eod = chambData.eod();
			    //int halfP = chambData.halfP();
			 } 
			 

			 if(typeOfRecord==RPCRecord::RMBDiscarded)
			 {
			    RMBErrorData  discarded(index);
			 }
			 if(typeOfRecord==RPCRecord::RMBCorrupted)
			 {
			    RMBErrorData  corrupted(index);
			 }
	
	
			 //Go to beginning of next record
			 theRecord.next();
			
			
			 }
			 
			 // Now check that next word is the Trailer as expected
		
		}
		
	}
       
        // Insert the new product in the event  
	//e.put(producedRPCDigis);
  
 
   
  
}

