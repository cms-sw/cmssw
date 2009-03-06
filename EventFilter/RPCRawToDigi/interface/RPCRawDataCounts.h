#ifndef EventFilter_RPCRawToDigi_RPCRawDataCounts_H
#define EventFilter_RPCRawToDigi_RPCRawDataCounts_H

#include <map>
#include <vector>

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
  void addRecordType(int fed, int type);
  void addReadoutError(int error);

private:
   std::map<int, std::vector<int> > theRecordTypes; 
   std::map<int,int> theReadoutErrors; 
};
#endif
