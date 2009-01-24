#include "DataFormats/RPCDigi/interface/ErrorRDM.h"

using namespace rpcrawtodigi;

bool ErrorRDM::matchType(const DataRecord & record)
{
  return ( RDM_TYPE_FLAG == (record.data() >> RDM_TYPE_SHIFT) ); 
}

unsigned int ErrorRDM::rmb() const
{
  return (theData & RMB_MASK);
}

std::string ErrorRDM::print() const
{
  std::ostringstream str;
  str <<" RDM,  rmb: "<< rmb();
  return str.str();
} 


