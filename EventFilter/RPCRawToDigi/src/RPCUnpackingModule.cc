/* 
 *
 *  
 *  
 */

#include <EventFilter/RPCRawToDigi/interface/RPCUnpackingModule.h>
#include <EventFilter/RPCRawToDigi/src/RPCDaqCMSFormatter.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>

using namespace raw;
using namespace edm;
using namespace std;

//#define SLINK_WORD_SIZE 8 //define it here?

#include <iostream>

RPCUnpackingModule::RPCUnpackingModule(const edm::ParameterSet& pset) : 
  unpacker(new RPCDaqCMSFormatter()) {
 
  produces<raw::RPCDigiCollection>();

}

RPCUnpackingModule::~RPCUnpackingModule(){
  delete unpacker;
}


void RPCUnpackingModule::produce(Event & e, const EventSetup& c){

  Handle<FEDRawDataCollection> allFEDRawData; 
  e.getByLabel("DaqRawData", allFEDRawData); //FED Raw data for all feds in the event


  auto_ptr<RPCDigiCollection> producedRPCDigis(new RPCDigiCollection);


    
	const std::pair<int,int> rpcFEDS=FEDNumbering::getMRpcFEDIds();
	for (int id= rpcFEDS.First(); id<=rpcFEDS.Second(); ++id){ 

		const FEDRawData & fedData = allFEDRawData->FEDData(id);
		
		
		if(fedData.size()){
		
      			const unsigned char* index = fedData.data();
			FEDHeader fedHeader(index); 
			
			const unsigned char* trailerIndex=index+feddata.size()
		 	FEDTrailer fedTrailer(trailerIndex);
			

// Beginning of RPC Records Unpacking
			index+=numberOfHeaders*SLINK_WORD_SIZE; //does this point to
								//beginning or end of first record?


			while( index != trailerIndex ){
			
			 RPCRecord theRecord(index);
			 
			 // Find what type of record it is
			 enum recordTypes typeOfRecord theRecord.type();
			 
			 if(typeOfRecord==RPCRecord::DataChamber)
			 {
			    RPCChamberData chambData(index);
			    int rpcChamber = chambData.chamberNumber();
			    int partitionNumber = chambData.partitionNumber();
			    int eod = chambData.eod();
			    int halfP = chambdata.halfP();
			 }
			 

			 //Go to beginning of next record
			 index=theRecord.next();
			
			
			 }
			 
			 // Now check that next word is the Trailer as expected
		
		}
		
	}
       
        // Insert the new product in the event  
	e.put(producedRPCDigis);
  
 
   
  
}

