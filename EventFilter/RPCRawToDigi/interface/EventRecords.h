#ifndef EventFilter_RPCRawToDigi_EventRecords_H
#define EventFilter_RPCRawToDigi_EventRecords_H

#include "EventFilter/RPCRawToDigi/interface/DataRecord.h" 
#include "EventFilter/RPCRawToDigi/interface/BXRecord.h" 
#include "EventFilter/RPCRawToDigi/interface/LBRecord.h" 
#include "EventFilter/RPCRawToDigi/interface/TBRecord.h" 
#include <vector>

namespace rpcrawtodigi {
class EventRecords {
public:

  EventRecords(int bx=0) 
    : theTriggerBX(bx), 
      theValidBX(false), theValidTB(false), theValidLB(false)
  {}

  EventRecords(int bx, const BXRecord & bxr, const TBRecord & tbr, const LBRecord & lbr)
    : theTriggerBX(bx),
      theValidBX(true), theValidTB(true), theValidLB(true),
      theBXRecord(bxr), theTBRecord(tbr), theLBRecord(lbr)
  {}

  void add(const DataRecord & record);

  int triggerBx() const { return theTriggerBX;}

  bool complete() const { return theValidBX && theValidTB && theValidLB; }

  bool samePartition(const EventRecords & r) const;

  const BXRecord & bxRecord() const { return theBXRecord; }
  const TBRecord & tbRecord() const { return theTBRecord; }
  const LBRecord & lbRecord() const { return theLBRecord; }

  static std::vector<EventRecords> mergeRecords(const std::vector<EventRecords> & r); 

private:
  int theTriggerBX;
  bool theValidBX, theValidTB, theValidLB; 
  BXRecord theBXRecord; 
  TBRecord theTBRecord;
  LBRecord theLBRecord;
};
}
#endif
