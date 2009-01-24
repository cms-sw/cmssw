#ifndef DataFormats_RPCDigi_RecordBX_H
#define DataFormats_RPCDigi_RecordBX_H

#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"

namespace rpcrawtodigi{
class RecordBX : public DataRecord {

private:
  static const int BX_TYPE_FLAG = 0xD;
  static const int BX_TYPE_SHIFT= 12;
  static const int BX_MASK  = 0xFFF;
  static const int BX_SHIFT = 0;

public:

  // empty record 
  RecordBX() : DataRecord() { }

  // set BX
  RecordBX(int bx) : DataRecord(0) {
    theData = (BX_TYPE_FLAG << BX_TYPE_SHIFT);
    theData |= (bx << BX_SHIFT);
  } 

  // specialize given recort to this type
  RecordBX(const DataRecord & rec) : DataRecord(rec) {}

  virtual ~RecordBX() {}
  int bx() const { return ((theData>>BX_SHIFT)&BX_MASK); } 
  std::string print()  const;
  static bool matchType(const DataRecord & record);
};
}
#endif
