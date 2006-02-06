/** \file
 * Implementation of class RPCUnpackingModule
 *
 *  $Date: 2006/02/02 16:25:51 $
 *  $Revision: 1.9 $
 *
 * \author Ilaria Segoni
 */

#include <EventFilter/RPCRawToDigi/interface/RPCUnpackingModule.h>
#include <EventFilter/RPCRawToDigi/interface/RPCUnpackingParameters.h>
#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>

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


#include <DataFormats/MuonDetId/interface/RPCDetId.h>

#include <iostream>

using namespace edm;
using namespace std;

RPCUnpackingModule::RPCUnpackingModule(const edm::ParameterSet& pset) 
:nEvents(0), currentBX(0), currentChn(0){
  
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
       int bx=0;
       vector<int> stripsOn;
       RPCDetId detId;
       
       while( index != trailerIndex ){
       
       
         
	 /// Loop on RPC Records
         int numOfRecords=0;
	 RPCRecord::recordTypes expectedRecord=RPCRecord::UndefinedType;
         
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
	  
	  if(typeOfRecord==RPCRecord::StartOfBXData)      {
	      bx= this->unpackBXRecord(recordIndex);
	  }	 
	  
	  
	  if(typeOfRecord==RPCRecord::StartOfChannelData) {
	       detId=this->unpackChannelRecord(recordIndex);
	  }
	  
	  
	  /// Unpacking Strips With Hit
	  if(typeOfRecord==RPCRecord::ChamberData)        {
	       stripsOn=this->unpackChamberRecord(recordIndex);
	  
	  
          for(std::vector<int>::iterator pStrip = stripsOn.begin(); pStrip !=
	                    stripsOn.end(); ++pStrip){
                int strip = *(pStrip);
          	  
                /// Creating RPC digi
	        RPCDigi digi(bx, strip);

	        /// Committing to the event
	        producedRPCDigis->insertDigi(detId,digi);
	  }
	  
	  
	  
	  }
	  
          if(typeOfRecord==RPCRecord::RMBDiscarded)       this->unpackRMBCorruptedRecord(recordIndex);
	  
          
	  if(typeOfRecord==RPCRecord::DCCDiscarded) rpcData.addDCCDiscarded();

	  
	  }
          
	  
          ///Go to beginning of next word
          index+= rpc::unpacking::SLINK_WORD_SIZE;


      }
		
      ///Send information to DQM
      if(instatiateDQM) monitor->process(rpcData);
   }
		
  }
       
        // Insert the new product in the event  
	//e.put(producedRPCDigis);
  
 
   
  
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




int RPCUnpackingModule::unpackBXRecord(const unsigned char* recordIndex) {

const unsigned int* recordIndexInt=reinterpret_cast<const unsigned int*>(recordIndex);

    RPCBXData bxData(recordIndexInt);
    if(printout) cout<<"Found BX record, BX= "<<bxData.bx()<<endl;
    return bxData.bx();
    rpcData.addBXData(bxData);

} 


RPCDetId RPCUnpackingModule::unpackChannelRecord(const unsigned char* recordIndex) {

const unsigned int* recordIndexInt=reinterpret_cast<const unsigned int*>(recordIndex);

    RPCChannelData chnData(recordIndexInt);
    if(printout) cout<<"Found start of Channel Data Record, Channel: "<< chnData.channel()<<
 	 " Readout/Trigger Mother Board: "<<chnData.tbRmb()<<endl;
    
    RPCDetId detId/*=chnData.detId()*/;
    return detId;
    
    rpcData.addChnData(chnData);

} 

vector<int> RPCUnpackingModule::unpackChamberRecord(const unsigned char* recordIndex) {

const unsigned int* recordIndexInt=reinterpret_cast<const unsigned int*>(recordIndex);

   RPCChamberData cmbData(recordIndexInt);
    if(printout) cout<< "Found Chamber Data, Chamber Number: "<<cmbData.chamberNumber()<<
 	" Partition Data "<<cmbData.partitionData()<<
 	" Half Partition " << cmbData.halfP()<<
 	" Data Truncated: "<<cmbData.eod()<<
 	" Partition Number " <<  cmbData.partitionNumber()
 	<<endl;
	
    vector<int> stripID/*=cmbData.getStrips()*/;
    return stripID;
    
    rpcData.addRPCChamberData(cmbData);
}



void RPCUnpackingModule::unpackRMBCorruptedRecord(const unsigned char* recordIndex) {



const unsigned int* recordIndexInt=reinterpret_cast<const unsigned int*>(recordIndex);

     RMBErrorData  discarded(recordIndexInt);
     rpcData.addRMBDiscarded(discarded);
     rpcData.addRMBCorrupted(discarded);

 }




DEFINE_FWK_MODULE(RPCUnpackingModule)
