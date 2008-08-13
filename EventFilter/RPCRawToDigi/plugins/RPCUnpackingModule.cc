#include "RPCUnpackingModule.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRawDataCounts.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "RPCReadOutMappingWithFastSearch.h"
#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"
#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"
#include "EventFilter/RPCRawToDigi/interface/DebugDigisPrintout.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"

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
  theCabling = new RPCReadOutMapping("");
  produces<RPCDigiCollection>();
  produces<RPCRawDataCounts>();
}

RPCUnpackingModule::~RPCUnpackingModule()
{ 
  delete theCabling;
}


void RPCUnpackingModule::produce(Event & ev, const EventSetup& es){

  static bool debug = edm::MessageDrop::instance()->debugEnabled;
  if (debug) LogDebug ("RPCUnpacker") <<"+++\nEntering RPCUnpackingModule::produce";
 
  Handle<FEDRawDataCollection> allFEDRawData; 
  ev.getByLabel(dataLabel_,allFEDRawData); 

  static edm::ESWatcher<RPCEMapRcd> recordWatcher;
  static RPCReadOutMappingWithFastSearch readoutMappingSearch;

  if (recordWatcher.check(es)) {  
    delete theCabling;
    ESHandle<RPCEMap> readoutMapping;
    if (debug) LogTrace("") << "record has CHANGED!!, initialise readout map!";
    es.get<RPCEMapRcd>().get(readoutMapping);
    theCabling = readoutMapping->convert();
    if (debug) LogTrace("") <<" READOUT MAP VERSION: " << theCabling->version() << endl;
    readoutMappingSearch.init(theCabling);
  }

  std::auto_ptr<RPCDigiCollection> producedRPCDigis(new RPCDigiCollection);
  std::auto_ptr<RPCRawDataCounts> producedRawDataCounts( new RPCRawDataCounts);

  std::pair<int,int> rpcFEDS=FEDNumbering::getRPCFEDIds();
 
  eventCounter_++; 
 
  if (debug) LogTrace ("") <<"Beginning To Unpack Event: "<<eventCounter_;

  int status = 0;
  for (int fedId= rpcFEDS.first; fedId<=rpcFEDS.second; ++fedId){  

    const FEDRawData & rawData = allFEDRawData->FEDData(fedId);
    RPCRecordFormatter interpreter(fedId, &readoutMappingSearch) ;
    int currentBX =0;
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
        producedRawDataCounts->addReadoutError(RPCRawDataCounts::HeaderCheckFail); 
        if (debug) LogTrace("") <<" ** PROBLEM **, header.check() failed, break"; 
        break; 
      }
      if ( fedHeader.sourceID() != fedId) {
        producedRawDataCounts->addReadoutError(RPCRawDataCounts::InconsitentFedId); 
        if (debug) LogTrace ("") <<" ** PROBLEM **, fedHeader.sourceID() != fedId"
            << "fedId = " << fedId<<" sourceID="<<fedHeader.sourceID(); 
      }
      currentBX = fedHeader.bxID();
      moreHeaders = fedHeader.moreHeaders();
      if (debug) {
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
        producedRawDataCounts->addReadoutError(RPCRawDataCounts::TrailerCheckFail);
        if (debug) LogTrace("") <<" ** PROBLEM **, trailer.check() failed, break";
        break;
      }
      if ( fedTrailer.lenght()!= nWords) {
        producedRawDataCounts->addReadoutError(RPCRawDataCounts::InconsistentDataSize); 
        if (debug) LogTrace("")<<" ** PROBLEM **, fedTrailer.lenght()!= nWords, break";
        break;
      }
      moreTrailers = fedTrailer.moreTrailers();
      if (debug) {
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
    if (debug) {
      ostringstream str;
      for (const Word64* word = header+1; word != trailer; word++) {
        str<<"    data: "<<*reinterpret_cast<const bitset<64>*>(word) << endl; 
      }
      LogTrace("") << str.str();
    }
    EventRecords event(currentBX);
    for (const Word64* word = header+1; word != trailer; word++) {
      for( int iRecord=1; iRecord<=4; iRecord++){
        typedef DataRecord::RecordType Record;
        const Record* pRecord = reinterpret_cast<const Record* >(word+1)-iRecord;
        DataRecord data(*pRecord);
        if (debug) LogTrace("")<<"record: " <<data.print()<<" record type:"<<data.type(); 
        event.add(data);
        producedRawDataCounts->addRecordType(fedId, data.type());
        int statusTMP = 0;
        if (event.complete()) statusTMP= 
            interpreter.recordUnpack(event, producedRPCDigis, producedRawDataCounts); 
        if (statusTMP != 0) status = statusTMP;
      }
    }
  }
  if (status) LogWarning(" RPCUnpackingModule - There was unpacking PROBLEM in this event");
  if (debug) LogTrace("") << DebugDigisPrintout()(producedRPCDigis.get()) << endl;
  ev.put(producedRPCDigis);  
  ev.put(producedRawDataCounts);
}
