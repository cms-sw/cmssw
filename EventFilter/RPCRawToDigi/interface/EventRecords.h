#ifndef EventFilter_RPCRawToDigi_EventRecords_H
#define EventFilter_RPCRawToDigi_EventRecords_H

#include "EventFilter/RPCRawToDigi/interface/DataRecord.h" 
#include "EventFilter/RPCRawToDigi/interface/RecordBX.h" 
#include "EventFilter/RPCRawToDigi/interface/RecordCD.h" 
#include "EventFilter/RPCRawToDigi/interface/RecordSLD.h" 
#include <vector>

namespace rpcrawtodigi {
class EventRecords {
public:

  EventRecords(int bx=0) 
    : theTriggerBX(bx), 
      theValidBX(false), theValidLN(false), theValidCD(false)
  {}

  EventRecords(int bx, const RecordBX & bxr, const RecordSLD & tbr, const RecordCD & lbr)
    : theTriggerBX(bx),
      theValidBX(true), theValidLN(true), theValidCD(true),
      theRecordBX(bxr), theRecordSLD(tbr), theRecordCD(lbr)
  {}

  void add(const DataRecord & record);

  int triggerBx() const { return theTriggerBX;}

  bool complete() const { return theValidBX && theValidLN && theValidCD; }

  bool hasErrors() const { return (theErrors.size()>0); }

  bool samePartition(const EventRecords & r) const;

  const RecordBX & recordBX() const { return theRecordBX; }
  const RecordSLD & recordSLD() const { return theRecordSLD; }
  const RecordCD & recordCD() const { return theRecordCD; }
  const std::vector<DataRecord> & errors() const { return theErrors; }

  static std::vector<EventRecords> mergeRecords(const std::vector<EventRecords> & r); 

  std::string print(DataRecord::recordName type) const;

private:
  int theTriggerBX;
  bool theValidBX, theValidLN, theValidCD; 
  RecordBX theRecordBX; 
  RecordSLD theRecordSLD;
  RecordCD theRecordCD;
  std::vector<DataRecord> theErrors;
};
}
#endif
