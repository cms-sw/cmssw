/** \file
 * Implementation of class RPCUnpackingModule
 *
 *  $Date: 2005/12/12 17:32:11 $
 *  $Revision: 1.6 $
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

using namespace edm;
using namespace std;


#include <iostream>

#define SLINK_WORD_SIZE 8
#define RPC_RECORD_SIZE 2


RPCUnpackingModule::RPCUnpackingModule(const edm::ParameterSet& pset)  
{
  nEvents=0;
  printout = pset.getUntrackedParameter<bool>("PrintOut", false); 
  hexprintout = pset.getUntrackedParameter<bool>("PrintHexDump", false); 
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
	  theRecord.recordUnpack(typeOfRecord);
	  
	  }
          
	  
          ///Go to beginning of next word
          index+=SLINK_WORD_SIZE;


        }
 // Now check that next word is the Trailer as expected
		
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





DEFINE_FWK_MODULE(RPCUnpackingModule)
