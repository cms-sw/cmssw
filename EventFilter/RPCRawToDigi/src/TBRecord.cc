#include "EventFilter/RPCRawToDigi/interface/TBRecord.h"
using namespace rpcrawtodigi;

std::string TBRecord::print() const 
{
  std::ostringstream str;
  str <<" rmb = "<<rmb(); 
  str <<" lnk = "<<tbLinkInputNumber();
  return str.str(); 
}

