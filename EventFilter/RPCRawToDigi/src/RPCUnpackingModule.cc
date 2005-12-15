/** \file
 * Implementation of class RPCUnpackingModule
 *
 *  $Date: 2005/12/14 13:35:22 $
 *  $Revision: 1.7 $
 *
 * \author Ilaria Segoni
 */

#include <EventFilter/RPCRawToDigi/interface/RPCUnpackingModule.h>
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

using namespace edm;
using namespace std;


#include <iostream>

#define SLINK_WORD_SIZE 8
#define RPC_RECORD_SIZE 2


RPCUnpackingModule::RPCUnpackingModule(const edm::ParameterSet& pset)  
{
  nEvents=0;
  currentBX=0;
  currentChn=0;
  
  printout = pset.getUntrackedParameter<bool>("PrintOut", false); 
  hexprintout = pset.getUntrackedParameter<bool>("PrintHexDump", false); 
  
  instatiateDQM = pset.getUntrackedParameter<bool>("runDQM", false);
  if(instatiateDQM){
   
   monitor = edm::Service<RPCMonitorInterface>().operator->(); 
  
  }
  
  
  
  produces<RPCDigiCollection>();

}

RPCUnpackingModule::~RPCUnpackingModule(){
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
      int numberOfHeaders= HeaderUnpacker(index);
 
      /// Unpack FED Trailer(s)
      const unsigned char* trailerIndex=index+fedData.size()-SLINK_WORD_SIZE;
      int numberOfTrailers=TrailerUnpacker(trailerIndex);
       
      if(printout) cout<<"Found "<<numberOfHeaders<<" Headers and "<<numberOfTrailers<<" Trailers"<<endl;		
  
      
      /// Beginning of RPC Records Unpacking
      index += numberOfHeaders*SLINK_WORD_SIZE; 
       
      /// Loop on S-LINK words 
       while( index != trailerIndex ){
       
         
	 /// Loop on RPC Records
         int numOfRecords=0;
         for(int nRecord=0; nRecord<4; nRecord++){
          
	  const unsigned char* recordIndex;
	  recordIndex=index+SLINK_WORD_SIZE-(nRecord+1)*RPC_RECORD_SIZE;
	   
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
	  recordUnpack(typeOfRecord,recordIndex);
	  
	  }
          
	  
          ///Go to beginning of next word
          index+=SLINK_WORD_SIZE;


        }
		
      ///Send information to DQM
      if(instatiateDQM) monitor->process(rpcData);
   }
		
  }
       
        // Insert the new product in the event  
	//e.put(producedRPCDigis);
  
 
   
  
}

int RPCUnpackingModule::HeaderUnpacker(const unsigned char* headerIndex){
 
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




int RPCUnpackingModule::TrailerUnpacker(const unsigned char* trailerIndex){
 
 int numberOfTrailers=0;
 bool moreTrailers = true;
 
 while(moreTrailers){
 
  FEDTrailer fedTrailer(trailerIndex);
  
  if(fedTrailer.check())  {
       
       numberOfTrailers++;	   
       trailerIndex -= SLINK_WORD_SIZE;     

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



void RPCUnpackingModule::recordUnpack(RPCRecord::recordTypes  type, const unsigned char* recordIndex){

const unsigned int* recordIndexInt=reinterpret_cast<const unsigned int*>(recordIndex);
/// BX Data type
 if(type==RPCRecord::StartOfBXData){
   
    RPCBXData bxData(recordIndexInt);
    if(printout) cout<<"Found BX record, BX= "<<bxData.bx()<<endl;
    currentBX=bxData.bx();
    rpcData.addBXData(bxData);

 } 

/// Start of Channel Data Type
 if(type==RPCRecord::StartOfChannelData){
   
    RPCChannelData chnData(recordIndexInt);
    if(printout) cout<<"Found start of Channel Data Record, Channel: "<< chnData.channel()<<
 	 " Readout/Trigger Mother Board: "<<chnData.tbRmb()<<endl;
    currentChn=chnData.channel();
    rpcData.addChnData(chnData);
 
 } 

/// Chamber Data 
 if(type==RPCRecord::ChamberData){
  
   RPCChamberData cmbData(recordIndexInt);
    if(printout) cout<< "Found Chamber Data, Chamber Number: "<<cmbData.chamberNumber()<<
 	" Partition Data "<<cmbData.partitionData()<<
 	" Half Partition " << cmbData.halfP()<<
 	" Data Truncated: "<<cmbData.eod()<<
 	" Partition Number " <<  cmbData.partitionNumber()
 	<<endl;
    rpcData.addRPCChamberData(cmbData);

 }

/// RMB Discarded
 if(type==RPCRecord::RMBDiscarded){
 
  RMBErrorData  discarded(recordIndexInt);
     rpcData.addRMBDiscarded(discarded);
     rpcData.addRMBCorrupted(discarded);

 }

/// DCC Discraded
 if(type==RPCRecord::DCCDiscarded){
     rpcData.addDCCDiscarded();
 }

 //delete recordIndexInt;
 //delete recordIndex;
 //recordIndexInt=0;
// recordIndex=0;

}


DEFINE_FWK_MODULE(RPCUnpackingModule)
