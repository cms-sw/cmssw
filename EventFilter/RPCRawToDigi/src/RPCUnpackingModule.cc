/** \file
 * Implementation of class RPCUnpackingModule
 *
 *  $Date: 2006/10/26 18:25:42 $
 *  $Revision: 1.23 $
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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/RPCRawToDigi/interface/RPCMonitorInterface.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/DataRecord/interface/RPCReadOutMappingRcd.h"


#include <iostream>
#include <bitset>

using namespace edm;
using namespace std;

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

  if(instatiateDQM) delete monitor;
  monitor = 0;

}


void RPCUnpackingModule::produce(Event & e, const EventSetup& c){

  edm::LogInfo ("RPCUnpacker") <<"+++\nEntering RPCUnpackingModule::produce";
 
 /// Get Data from all FEDs
  Handle<FEDRawDataCollection> allFEDRawData; 
  e.getByType(allFEDRawData); 

  edm::ESHandle<RPCReadOutMapping> readoutMapping;
  c.get<RPCReadOutMappingRcd>().get(readoutMapping);

  edm::LogInfo ("RPCUnpacker") <<"Got FEDRawData";
 
  std::auto_ptr<RPCDigiCollection> producedRPCDigis(new RPCDigiCollection);

  std::pair<int,int> rpcFEDS=FEDNumbering::getRPCFEDIds();
 
  nEvents++; 
 
  edm::LogInfo ("RPCUnpacker") <<"Beginning To Unpack Event: "<<nEvents;

  for (int id= rpcFEDS.first; id<=rpcFEDS.second; ++id){  

    const FEDRawData & fedData = allFEDRawData->FEDData(id);
    RPCRecordFormatter interpreter(id, readoutMapping.product()) ;
    RPCFEDData rpcRawData;
  
    edm::LogInfo ("RPCUnpacker") <<"Beginning to Unpack FED number "<<id<<", FED size: "<<fedData.size()<<" bytes";			 

    if(fedData.size()){

      const unsigned char* index = fedData.data();

      {
        ostringstream str;
        str <<"  header: "<< *reinterpret_cast<const bitset<64>*> (index) << endl;
        FEDHeader header(index);
        str <<"  header triggerType: " << header.triggerType()<<endl;
        str <<"  header lvl1ID:      " << header.lvl1ID() << endl;
        str <<"  header bxID:        " << header.bxID() << endl;
        str <<"  header sourceID:    " << header.sourceID() << endl;       
        str <<"  header version:    " << header.version() << endl;       
        for (unsigned int idata = 1; idata <= ( fedData.size()/8-2); idata ++) {
          str<<"    data: "<<*reinterpret_cast<const bitset<64>*> (index+idata*8) << endl; 
        }
        str <<" trailer: "<<  *reinterpret_cast<const bitset<64>*> (index+fedData.size()-8); 
        FEDTrailer trailer(index+fedData.size()-8);
        str <<"  trailer lenght:    "<< trailer.lenght()<<endl;
        str <<"  trailer crc:       "<<trailer.crc()<<endl;
        str <<"  trailer evtStatus: "<<trailer.evtStatus()<<endl;
        str <<"  trailer ttsBits:   "<<trailer.ttsBits()<<endl;
        
        edm::LogInfo ("RPCUnpacker FED data:") << str.str();
      }
       
      /// Unpack FED Header(s)
      int numberOfHeaders= this->unpackHeader(index, rpcRawData);
      int currentBX = rpcRawData.fedHeaders()[numberOfHeaders-1].bxID(); 
 
      /// Unpack FED Trailer(s)
      const unsigned char* trailerIndex=index+fedData.size()- rpc::unpacking::SLINK_WORD_SIZE;
      int ttsSTatus=this->unpackTrailer(trailerIndex, rpcRawData);
       
      if(ttsSTatus){
        edm::LogError ("RPCUnpacker") <<"ERROR REPORTED FROM TTS Status= "<< ttsSTatus;
//        break;
      }

     				
      /// Beginning of RPC Records Unpacking
      index += numberOfHeaders* rpc::unpacking::SLINK_WORD_SIZE; 
       
      /// Loop on S-LINK words        
      while( index != trailerIndex ){            
       
	  /// Loop on RPC Records
        int numOfRecords=0;
	  enum RPCRecord::recordTypes previousRecord=RPCRecord::UndefinedType;

	  for( int nRecord=0; nRecord<4; nRecord++){
          
	    const unsigned char* recordIndex = index+ rpc::unpacking::SLINK_WORD_SIZE -(nRecord+1)* rpc::unpacking::RPC_RECORD_SIZE;

          edm::LogInfo ("RPCUnpacker RECORD: ")<<"===> INTERPRET RECORD: " <<*reinterpret_cast<const bitset<16>*>(recordIndex);
#ifdef DEBUG_RPCUNPACKER	  
	    numOfRecords++;
	    const unsigned int* word=reinterpret_cast<const unsigned int*>(recordIndex);
	    std::cout<<oct<<*word<<" ";
	    if(numOfRecords==4) { cout<<std::endl; }		
#endif	
	    RPCRecord theRecord(recordIndex,printout,previousRecord);
        
          /// Find out type of record
          RPCRecord::recordTypes typeOfRecord = theRecord.computeType();					

	    /// Check Record is of expected type
	    bool missingRecord = theRecord.check();
	    /// Unpack the Record 	  
	    try{
            interpreter.recordUnpack(theRecord,producedRPCDigis,rpcRawData,currentBX);
          }
          catch (cms::Exception & e) { LogError("Exception catched, skip digi")<<e.what(); }		  
	  
        } 
          
        ///Go to beginning of next word
        index+= rpc::unpacking::SLINK_WORD_SIZE;
      }
		
      ///Send information to DQM
      if(instatiateDQM) monitor->process(rpcRawData);
		
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
        edm::LogInfo ("RPCUnpacker") <<"Found "<< numberOfHeaders<<" Headers";
  
 return numberOfHeaders;    		

}




int RPCUnpackingModule::unpackTrailer(const unsigned char* trailerIndex, RPCFEDData & rawData) {
 
 int numberOfTrailers=0;
 int tts=0;
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
        
	if(fedTrailer.ttsBits()) tts=fedTrailer.ttsBits();
  
  }else{
       moreTrailers=false;
  }

 }
 
        edm::LogInfo ("RPCUnpacker") <<"Found "<< numberOfTrailers<<" Trailers";
 
 return tts;
}
