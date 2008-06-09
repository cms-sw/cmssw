#ifndef EventFilter_RPCRawToDigi_LBRecord_H
#define EventFilter_RPCRawToDigi_LBRecord_H

#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"
#include "EventFilter/RPCRawToDigi/interface/RPCLinkBoardData.h"

namespace rpcrawtodigi{
class LBRecord : public DataRecord {

private:
  static const int PARTITION_DATA_MASK  = 0XFF;
  static const int PARTITION_DATA_SHIFT =0;

  static const int PARTITION_NUMBER_MASK = 0XF;
  static const int PARTITION_NUMBER_SHIFT =10;

  static const int HALFP_MASK = 0X1;
  static const int HALFP_SHIFT =8;

  static const int EOD_MASK = 0X1;
  static const int EOD_SHIFT =9;

  static const int LB_MASK = 0X3;
  static const int LB_SHIFT =14;

  static const int BITS_PER_PARTITION=8; 

public:

  // empty record 
  LBRecord() : DataRecord() { }

  // set LB
  LBRecord(const RPCLinkBoardData & lbData);

  // set LB from raw
  LBRecord(RecordType lbData);

  // specialize given recort to this type
  LBRecord(const DataRecord & rec) : DataRecord(rec) {}

  virtual ~LBRecord() {}

  RPCLinkBoardData lbData() const;
};
}
#endif
