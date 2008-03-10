#ifndef EventFilter_RPCRawToDigi_ErrorSDDM_H
#define EventFilter_RPCRawToDigi_ErrorSDDM_H

#include <bitset>
#include <string> 
#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"

namespace rpcrawtodigi {

class ErrorSDDM : public DataRecord {
private:
  static const unsigned int SDDM_TYPE_FLAG = 0xE801;  // 1110100000000001 
public:
  ErrorSDDM(const DataRecord & r = SDDM_TYPE_FLAG) : DataRecord(r) {}
  static bool matchType(const DataRecord & record) { return ( SDDM_TYPE_FLAG == record.data() ); }
  std::string print() const { return " SDDM "; }
};

}

#endif
