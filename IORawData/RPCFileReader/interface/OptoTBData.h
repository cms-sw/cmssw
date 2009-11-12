#ifndef IORawData_RPCFileReader_OptoTBData_H
#define IORawData_RPCFileReader_OptoTBData_H

#include <vector>
#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"

class OptoTBData {
public:
   struct LMD { LMD(){};
                LMD(unsigned int raw);
                unsigned int raw() const;
                unsigned int dat, del, eod, hp, lb, par;
                bool operator<(const LMD &) const; }; 
   OptoTBData(){}
   OptoTBData(unsigned int fedId, const rpcrawtodigi::EventRecords & event);
   bool operator<(const OptoTBData & a) const;
   virtual ~OptoTBData(){}
   unsigned int bx() const { return theBX; }
   unsigned int tc() const { return theTC; }
   unsigned int tb() const { return theTB; }
   unsigned int ol() const { return theOL; }
   const LMD & lmd() const { return theLMD; } 
private:
  static std::pair<int,int> getTCandTBNumbers(unsigned int rmb, unsigned int dcc);
private:
  unsigned int theBX,theTC,theTB,theOL;
  LMD theLMD;
};
#endif
