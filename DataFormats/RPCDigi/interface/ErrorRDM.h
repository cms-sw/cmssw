#ifndef DataFormats_RPCDigi_ErrorRDM_H
#define DataFormats_RPCDigi_ErrorRDM_H

#include <bitset>
#include <string>
#include "DataFormats/RPCDigi/interface/DataRecord.h"

namespace rpcrawtodigi {

  class ErrorRDM : public DataRecord {
  public:
    ErrorRDM(const DataRecord r) : DataRecord(r) {}

    static bool matchType(const DataRecord& record);
    std::string print() const;

    unsigned int rmb() const;

  private:
    static const unsigned int RDM_TYPE_FLAG = 0x3A1;  // 1110100001
    static const unsigned int RDM_TYPE_SHIFT = 6;
    static const unsigned int RMB_MASK = 0x3F;  // 111111
  };

}  // namespace rpcrawtodigi

#endif
