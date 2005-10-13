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
  e.getByLabel("DaqRawData", allFEDRawData);


  auto_ptr<RPCDigiCollection> producedRPCDigis(new RPCDigiCollection);

  try{
    
	const std::pair<int,int> rpcFEDS=FEDNumbering::getMRpcFEDIds() throw 1;  
	for (int id= rpcFEDS.First(); id<=rpcFEDS.Second(); ++id){ 

		const FEDRawData & data = allFEDRawData->FEDData(id);
		if (data.size()){             
      
			unpacker->interpretRawData(data, *producedRPCDigis);
    
		}
	}
       
        // Insert the new product in the event  
	e.put(producedRPCDigis);
  
  }
 
  catch(int i) {

	if (i==1){
		cout<<" No FED id defined for RPC, Cannot unpack data"<<endl;
	}    
  }
	
  
  
}

