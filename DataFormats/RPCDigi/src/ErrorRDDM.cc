#include "DataFormats/RPCDigi/interface/ErrorRDDM.h"

using namespace rpcrawtodigi;

bool ErrorRDDM::matchType(const DataRecord& record) {
  return (RDDM_TYPE_FLAG == static_cast<unsigned int>(record.data() >> RDDM_TYPE_SHIFT));
}

unsigned int ErrorRDDM::rmb() const { return ((theData >> RMB_SHIFT) & RMB_MASK); }

unsigned int ErrorRDDM::link() const { return (theData & LNK_MASK); }

std::string ErrorRDDM::print() const {
  std::ostringstream str;
  str << " RDDM, rmb: " << rmb() << " lnk: " << link();
  return str.str();
}
