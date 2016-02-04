#ifndef EventFilter_RPCRawToDigi_debugDigisPrintout_h
#define EventFilter_RPCRawToDigi_debugDigisPrintout_h

#include <string>
#include <sstream>
#include <vector>
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

namespace rpcrawtodigi {
  class DebugDigisPrintout {

    struct MyDigi { 
       uint32_t det; int strip; int bx; 
       bool operator==(const MyDigi&o) const {
         return (det==o.det && strip==o.strip && bx==o.bx);
       } 
       bool operator< (const MyDigi&o) const { 
         if (this->det < o.det) return true;
         if (this->det > o.det) return false; 
         if (this->strip < o.strip) return true;
         return false;
       }
    };

  public:
    std::string operator()(const RPCDigiCollection * digis) {
      std::ostringstream str;
      str << "DebugDigisPrintout:";
      if (!digis) return str.str();
      typedef  DigiContainerIterator<RPCDetId, RPCDigi> DigiRangeIterator;
      std::vector<MyDigi> myDigis;

      int nDet = 0;
      int nDigisAll = 0;
      for (DigiRangeIterator it=digis->begin(); it != digis->end(); it++) {
        nDet++;
        RPCDetId rpcDetId = (*it).first;
        uint32_t rawDetId = rpcDetId.rawId();
        RPCDigiCollection::Range range = digis->get(rpcDetId);
        for (std::vector<RPCDigi>::const_iterator  id = range.first; id != range.second; id++) {
          nDigisAll++;
          const RPCDigi & digi = (*id);
          MyDigi myDigi = { rawDetId, digi.strip(), digi.bx() };
          if (myDigis.end() == std::find(myDigis.begin(), myDigis.end(), myDigi)) 
              myDigis.push_back(myDigi);
        } 
      }
      std::sort(myDigis.begin(),myDigis.end());
      str << " dets: "<<nDet<<" allDigis: "<<nDigisAll<<" unigueDigis: "<<myDigis.size()<<std::endl;
      for (std::vector<MyDigi>::const_iterator it = myDigis.begin(); it != myDigis.end(); ++it)
           str << "debugDIGI: "<< it->det<<", "<<it->strip<<", "<<it->bx<<std::endl;
      return str.str();
    }
  };
}
#endif

