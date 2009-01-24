#ifndef DataFormats_RPCDigi_ErrorRDDM_H
#define DataFormats_RPCDigi_ErrorRDDM_H

#include <bitset>
#include <string> 
#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"

namespace rpcrawtodigi {

class ErrorRDDM : public DataRecord {
public:

  ErrorRDDM(const DataRecord r) : DataRecord(r) {}

  static bool matchType(const DataRecord & record);
  std::string print() const;

  unsigned int rmb() const;
  unsigned int link() const;

private:
  static const unsigned int RDDM_TYPE_FLAG = 0x1E;  // 11110 
  static const unsigned int RDDM_TYPE_SHIFT = 11; 

  static const unsigned int RMB_MASK  = 0x3F;  // 111111
  static const unsigned int RMB_SHIFT = 5; 

  static const unsigned int LNK_MASK = 0x1F;   //11111 
};

}

#endif
