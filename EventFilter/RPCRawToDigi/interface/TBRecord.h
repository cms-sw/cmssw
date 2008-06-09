#ifndef EventFilter_RPCRawToDigi_TBRecord_H
#define EventFilter_RPCRawToDigi_TBRecord_H

#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"

namespace rpcrawtodigi{
class TBRecord : public DataRecord {

private:
  static const int TB_LINK_INPUT_NUMBER_MASK  = 0X1F;
  static const int TB_LINK_INPUT_NUMBER_SHIFT =0;
  static const int TB_RMB_MASK = 0X3F;
  static const int TB_RMB_SHIFT =5;

public:

  // empty record 
  TBRecord() : DataRecord() { }

  // set TB
  TBRecord(int tbLinkInputNumber, int rmb) : DataRecord(0) {
    theData = (controlWordFlag << RECORD_TYPE_SHIFT);
    theData |= (StartOfLBInputDataFlag << CONTROL_TYPE_SHIFT);
    theData |= (tbLinkInputNumber << TB_LINK_INPUT_NUMBER_SHIFT);
    theData |= (rmb << TB_RMB_SHIFT);
  } 

  // specialize given recort to this type
  TBRecord(const DataRecord & rec) : DataRecord(rec) {}

  virtual ~TBRecord() {}

  int tbLinkInputNumber() const {
     return (theData >> TB_LINK_INPUT_NUMBER_SHIFT)& TB_LINK_INPUT_NUMBER_MASK;
  }

  int rmb() const { return (theData >> TB_RMB_SHIFT) & TB_RMB_MASK; }

};
}
#endif
