#include "DataFormats/RPCDigi/interface/RecordBX.h"
using namespace rpcrawtodigi;

bool RecordBX::matchType(const DataRecord & record)
{
  return ( BX_TYPE_FLAG == (record.data() >> BX_TYPE_SHIFT) );
}


std::string RecordBX::print() const 
{
  std::ostringstream str; 
  str <<" BX,    bx = "<<bx(); 
  return str.str(); 
}
