#include "EventFilter/RPCRawToDigi/interface/BXRecord.h"
using namespace rpcrawtodigi;

std::string BXRecord::print() const 
{
  std::ostringstream str; 
  str <<" bx = "<<bx(); 
  return str.str(); 
}
