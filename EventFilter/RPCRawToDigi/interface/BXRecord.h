#ifndef EventFilter_RPCRawToDigi_BXRecord_H
#define EventFilter_RPCRawToDigi_BXRecord_H

#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"

namespace rpcrawtodigi{
class BXRecord : public DataRecord {

private:
  static const int BX_MASK  = 0XFFF;
  static const int BX_SHIFT = 0;

public:

  // empty record 
  BXRecord() : DataRecord() { }

  // set BX
  BXRecord(int bx) : DataRecord(0) {
    theData =  (controlWordFlag << RECORD_TYPE_SHIFT);
    theData |= (BXFlag << BX_TYPE_SHIFT);
    theData |= (bx << BX_SHIFT);
  } 

  // specialize given recort to this type
  BXRecord(const DataRecord & rec) : DataRecord(rec) {}

  virtual ~BXRecord() {}
  int bx() const { return ((theData>>BX_SHIFT)&BX_MASK); } 
};
}
#endif
