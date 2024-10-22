#include "DataFormats/RPCDigi/interface/RecordSLD.h"
using namespace rpcrawtodigi;

std::string RecordSLD::print() const {
  std::ostringstream str;
  str << " SLD,   rmb = " << rmb();
  str << " lnk = " << tbLinkInputNumber();
  return str.str();
}
