#ifndef DataFormats_RPCDigi_ErrorRCDM_H
#define DataFormats_RPCDigi_ErrorRCDM_H

#include <bitset>
#include <string>
#include "DataFormats/RPCDigi/interface/DataRecord.h"

namespace rpcrawtodigi {

  class ErrorRCDM : public DataRecord {
  public:
    ErrorRCDM(const DataRecord r) : DataRecord(r) {}

    static bool matchType(const DataRecord& record);
    std::string print() const;

    unsigned int rmb() const;
    unsigned int link() const;

  private:
    static const unsigned int RCDM_TYPE_FLAG = 0x1C;  // 11100
    static const unsigned int RCDM_TYPE_SHIFT = 11;

    static const unsigned int RMB_MASK = 0x3F;  // 111111
    static const unsigned int RMB_SHIFT = 5;

    static const unsigned int LNK_MASK = 0x1F;  //11111
  };

}  // namespace rpcrawtodigi

#endif
