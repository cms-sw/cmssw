#include "RawToXML.h"

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "DataFormats/RPCDigi/interface/DataRecord.h"
#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"

#include "IORawData/RPCFileReader/interface/XMLDataIO.h"
#include "IORawData/RPCFileReader/interface/OptoTBData.h"

#include "boost/bind.hpp"
#include <bitset>
#include <string>
#include <vector>

typedef uint64_t Word64;
using namespace rpcrawtodigi;


RawToXML::RawToXML(const edm::ParameterSet& conf)
  : theDataLabel(conf.getParameter<edm::InputTag>("InputLabel"))
{
  std::string outFileName =  conf.getParameter<std::string>("xmlFileName");
  theWriter = new XMLDataIO(outFileName);
  
}

RawToXML::~RawToXML()
{
  delete theWriter;
}


void RawToXML::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  std::vector<OptoTBData> optoData; 
//  static bool debug = edm::MessageDrop::instance()->debugEnabled;
//  if (debug) LogDebug (" RawToXMLConvert") <<"Beginning To Unpack Event: ";

  edm::Handle<FEDRawDataCollection> allFEDRawData;
  ev.getByLabel(theDataLabel,allFEDRawData);
  for (int fedId= FEDNumbering::MINRPCFEDID; fedId<=FEDNumbering::MAXRPCFEDID; ++fedId){

    const FEDRawData & rawData = allFEDRawData->FEDData(fedId);
    int triggerBX =0;
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
      triggerBX = fedHeader.bxID();
      moreHeaders = fedHeader.moreHeaders();
//      if (debug) {
//        std::stringstream str;
//        str <<"  header: "<< *reinterpret_cast<const std::bitset<64>*> (header) << std::endl;
//        str <<"  header triggerType: " << fedHeader.triggerType()<<std::endl;
//        str <<"  header lvl1ID:      " << fedHeader.lvl1ID() << std::endl;
//        str <<"  header bxID:        " << fedHeader.bxID() << std::endl;
//        str <<"  header sourceID:    " << fedHeader.sourceID() << std::endl;
//        str <<"  header version:     " << fedHeader.version() << std::endl;
//        LogTrace("") << str.str();
//      }
    }


    //
    // check trailers
    //
    const Word64* trailer=reinterpret_cast<const Word64* >(rawData.data())+(nWords-1); trailer++;
    bool moreTrailers = true;
    while (moreTrailers) {
      trailer--;
      FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer));
      moreTrailers = fedTrailer.moreTrailers();
//      if (debug) {
//        std::stringstream str;
//        str <<" trailer: "<<  *reinterpret_cast<const std::bitset<64>*> (trailer) << std::endl;
//        str <<"  trailer lenght:    "<<fedTrailer.lenght()<<std::endl;
//        str <<"  trailer crc:       "<<fedTrailer.crc()<<std::endl;
//        str <<"  trailer evtStatus: "<<fedTrailer.evtStatus()<<std::endl;
//        str <<"  trailer ttsBits:   "<<fedTrailer.ttsBits()<<std::endl;
//        LogTrace("") << str.str();
//      }
    }

//    if (debug) {
//      std::stringstream str;
//      for (const Word64* word = header+1; word != trailer; word++) {
//        str<<"    data: "<<*reinterpret_cast<const std::bitset<64>*>(word) << std::endl;
//      }
//      LogTrace("") << str.str();
//    }
    EventRecords event(triggerBX);
    for (const Word64* word = header+1; word != trailer; word++) {
      for( int iRecord=1; iRecord<=4; iRecord++){
        const DataRecord::Data* pRecord = reinterpret_cast<const DataRecord::Data* >(word+1)-iRecord;
        DataRecord record(*pRecord);
        event.add(record);
        if (event.complete())optoData.push_back(OptoTBData(fedId,event));
//        if (debug) {
//          std::stringstream str;
//          str <<"record: "<<record.print()<<" hex: "<<std::hex<<*pRecord<<std::dec;
//          str <<" type:"<<record.type()<<DataRecord::print(record);
//          if (event.complete()) {
//            str<< " --> dccId: "<<fedId
//               << " bx:  "<<event.recordBX().bx()
//               << " rmb: " <<event.recordSLD().rmb()
//               << " lnk: "<<event.recordSLD().tbLinkInputNumber()
//               << " lb: "<<event.recordCD().lbInLink()
//               << " part: "<<event.recordCD().partitionNumber()
//               << " data: "<<event.recordCD().partitionData()
//               << " eod: "<<event.recordCD().eod();
//          }
//          LogTrace("") << str.str();
//        }
      }
    }
  }
  std::sort(optoData.begin(),optoData.end());
  theWriter->write(ev,optoData);
}

