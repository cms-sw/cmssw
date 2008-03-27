/** \file
 * Implementation of class RPCUnpackingModule
 *
 *  $Date: 2008/01/22 19:12:35 $
 *  $Revision: 1.3 $
 *
 * \author Ilaria Segoni
 */



#include "RPCUnpackingModule.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
//#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
//#include "CondFormats/DataRecord/interface/RPCReadOutMappingRcd.h"
#include "RPCReadOutMappingWithFastSearch.h"
#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"
#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"

#include <iostream>
#include <bitset>

using namespace edm;
using namespace std;
using namespace rpcrawtodigi;

typedef uint64_t Word64;


RPCUnpackingModule::RPCUnpackingModule(const edm::ParameterSet& pset) 
  : dataLabel_(pset.getUntrackedParameter<edm::InputTag>("InputLabel",edm::InputTag("source"))),
    eventCounter_(0)
{
  RPCCabling = new RPCReadOutMapping("");
  produces<RPCDigiCollection>();
}

RPCUnpackingModule::~RPCUnpackingModule(){ }


void RPCUnpackingModule::produce(Event & e, const EventSetup& c){

  edm::LogInfo ("RPCUnpacker") <<"+++\nEntering RPCUnpackingModule::produce";
 
 /// Get Data from all FEDs
  Handle<FEDRawDataCollection> allFEDRawData; 
  e.getByLabel(dataLabel_,allFEDRawData); 

//  edm::ESHandle<RPCReadOutMapping> readoutMapping;
//  c.get<RPCReadOutMappingRcd>().get(readoutMapping);
  edm::ESHandle<RPCEMap> readoutMapping;
  c.get<RPCEMapRcd>().get(readoutMapping);
  const RPCEMap* eMap=readoutMapping.product();

  if (eMap->theVersion != RPCCabling->version()) {
    delete RPCCabling;
    RPCCabling = eMap->convert();
  }

  static RPCReadOutMappingWithFastSearch readoutMappingSearch;
//  readoutMappingSearch.init(readoutMapping.product());
  readoutMappingSearch.init(RPCCabling);

  edm::LogInfo ("RPCUnpacker") <<"Got FEDRawData";
 
  std::auto_ptr<RPCDigiCollection> producedRPCDigis(new RPCDigiCollection);


  std::pair<int,int> rpcFEDS=FEDNumbering::getRPCFEDIds();
 
  eventCounter_++; 
 
  edm::LogInfo ("RPCUnpacker") <<"Beginning To Unpack Event: "<<eventCounter_;

  for (int fedId= rpcFEDS.first; fedId<=rpcFEDS.second; ++fedId){  

    const FEDRawData & rawData = allFEDRawData->FEDData(fedId);
    //RPCRecordFormatter interpreter(fedId, readoutMapping.product()) ;
    RPCRecordFormatter interpreter(fedId, &readoutMappingSearch) ;
    int currentBX =0;
    try {
      int nWords = rawData.size()/sizeof(Word64);
      if (nWords==0) continue;

      //
      // check headers
      //
      const Word64* header = reinterpret_cast<const Word64* >(rawData.data()); header--;
      bool moreHeaders = true;
      while (moreHeaders) {
        header++;
        FEDHeader fedHeader( reinterpret_cast<const unsigned char*>(header));
        if (!fedHeader.check()) {
          LogError(" ** PROBLEM **, header.check() failed, break"); 
          break; 
        }
        if ( fedHeader.sourceID() != fedId) {
          LogError(" ** PROBLEM **, fedHeader.sourceID() != fedId")
              << "fedId = " << fedId<<" sourceID="<<fedHeader.sourceID(); 
        }
        currentBX = fedHeader.bxID();
        moreHeaders = fedHeader.moreHeaders();
        {
          ostringstream str;
          str <<"  header: "<< *reinterpret_cast<const bitset<64>*> (header) << endl;
          str <<"  header triggerType: " << fedHeader.triggerType()<<endl;
          str <<"  header lvl1ID:      " << fedHeader.lvl1ID() << endl;
          str <<"  header bxID:        " << fedHeader.bxID() << endl;
          str <<"  header sourceID:    " << fedHeader.sourceID() << endl;
          str <<"  header version:     " << fedHeader.version() << endl;
          LogTrace("") << str.str();
        }
      }

      //
      // check trailers
      //
      const Word64* trailer=reinterpret_cast<const Word64* >(rawData.data())+(nWords-1); trailer++;
      bool moreTrailers = true;
      while (moreTrailers) {
        trailer--;
        FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer));
        if ( !fedTrailer.check()) {
          LogError(" ** PROBLEM **, trailer.check() failed, break");
          break;
        }
        if ( fedTrailer.lenght()!= nWords) {
          LogError(" ** PROBLEM **, fedTrailer.lenght()!= nWords, break");
          break;
        }
        moreTrailers = fedTrailer.moreTrailers();
        {
          ostringstream str;
          str <<" trailer: "<<  *reinterpret_cast<const bitset<64>*> (trailer) << endl; 
          str <<"  trailer lenght:    "<<fedTrailer.lenght()<<endl;
          str <<"  trailer crc:       "<<fedTrailer.crc()<<endl;
          str <<"  trailer evtStatus: "<<fedTrailer.evtStatus()<<endl;
          str <<"  trailer ttsBits:   "<<fedTrailer.ttsBits()<<endl;
          LogTrace("") << str.str();
        }
      }

      //
      // data records
      //
      {
        ostringstream str;
        for (const Word64* word = header+1; word != trailer; word++) {
          str<<"    data: "<<*reinterpret_cast<const bitset<64>*>(word) << endl; 
        }
        LogTrace("") << str.str();
      }
      EventRecords event(currentBX);
      int status = 0;
      for (const Word64* word = header+1; word != trailer; word++) {
	  for( int iRecord=1; iRecord<=4; iRecord++){
          typedef DataRecord::RecordType Record;
          const Record* pRecord = reinterpret_cast<const Record* >(word+1)-iRecord;
          DataRecord data(*pRecord);
          LogTrace("")<<"record: " <<data.print()<<" record type:"<<data.type(); 
          event.add(data);
          if (event.complete()) status = interpreter.recordUnpack(event, producedRPCDigis); 
        }
      }
    }
    catch ( cms::Exception & err) { LogError("RPCUnpacker exception") <<err.what(); }
  }
  e.put(producedRPCDigis);  
}
