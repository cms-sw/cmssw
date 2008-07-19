#ifndef EventFilter_RPCRawToDigi_RPCRawDataCounts_H
#define EventFilter_RPCRawToDigi_RPCRawDataCounts_H

#include <map>
#include <vector>
#include <string>
#include "TH1D.h"

class RPCRawDataCounts {
public:
  enum ReadoutError { NoProblem = 0,
         HeaderCheckFail = 1,
         InconsitentFedId = 2,
         TrailerCheckFail = 3,
         InconsistentDataSize = 4,  
         InvalidLB = 5,
         EmptyPackedStrips = 6,
         InvalidDetId = 7,
         InvalidStrip = 8 };
  RPCRawDataCounts() {}
  ~RPCRawDataCounts() { }
  void addRecordType(int fed, int type, int weight=1);
  void addReadoutError(int error, int weight=1);
  void operator+= (const RPCRawDataCounts& );
  std::string print() const;
  TH1D recordTypeHisto(int fedid) const;
  TH1D readoutErrorHisto() const;
private:
   std::map<int, std::vector<int> > theRecordTypes; 
   std::map<int,int> theReadoutErrors; 
};
#endif
