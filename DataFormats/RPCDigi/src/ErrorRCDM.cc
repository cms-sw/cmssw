#include "DataFormats/RPCDigi/interface/ErrorRCDM.h"

using namespace rpcrawtodigi;

bool ErrorRCDM::matchType(const DataRecord & record)
{
  return ( RCDM_TYPE_FLAG == (unsigned int)(record.data() >> RCDM_TYPE_SHIFT) ); 
}

unsigned int ErrorRCDM::rmb() const
{
  return ((theData >> RMB_SHIFT) & RMB_MASK);
}

unsigned int ErrorRCDM::link() const
{
  return (theData & LNK_MASK);
}


std::string ErrorRCDM::print() const
{
  std::ostringstream str;
  str <<" RCDM,  rmb: "<< rmb() <<" lnk: "<<link();
  return str.str();
} 


