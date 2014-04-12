#ifndef EventFilter_RPCRawToDigi_EventRecords_H
#define EventFilter_RPCRawToDigi_EventRecords_H

#include "DataFormats/RPCDigi/interface/DataRecord.h" 
#include "DataFormats/RPCDigi/interface/RecordBX.h" 
#include "DataFormats/RPCDigi/interface/RecordCD.h" 
#include "DataFormats/RPCDigi/interface/RecordSLD.h" 
#include <vector>

namespace rpcrawtodigi {
class EventRecords {
public:

  EventRecords(int triggerbx=0) 
    : theTriggerBX(triggerbx), 
      theValidBX(false), theValidLN(false), theValidCD(false)
  {}

  EventRecords(int bx, const RecordBX & bxr, const RecordSLD & tbr, const RecordCD & lbr)
    : theTriggerBX(bx),
      theValidBX(true), theValidLN(true), theValidCD(true),
      theRecordBX(bxr), theRecordSLD(tbr), theRecordCD(lbr)
  {}

  void add(const DataRecord & record);

  int triggerBx() const { return theTriggerBX;}

  int dataToTriggerDelay() const; 

  bool complete() const { return theValidBX && theValidLN && theValidCD; }

  bool hasErrors() const { return (theErrors.size()>0); }

  bool samePartition(const EventRecords & r) const;

  const RecordBX & recordBX() const { return theRecordBX; }
  const RecordSLD & recordSLD() const { return theRecordSLD; }
  const RecordCD & recordCD() const { return theRecordCD; }
  const std::vector<DataRecord> & errors() const { return theErrors; }

  static std::vector<EventRecords> mergeRecords(const std::vector<EventRecords> & r); 

  std::string print(const DataRecord::DataRecordType& type) const;

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
