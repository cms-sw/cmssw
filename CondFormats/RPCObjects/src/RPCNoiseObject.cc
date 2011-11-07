#include "CondFormats/RPCObjects/interface/RPCNoiseObject.h"
#include "FWCore/Utilities/interface/Exception.h"

std::vector<RPCNoiseObject::NoiseObjectItem>  const & 
RPCNoiseObject::getCls() const
{
  return v_cls;
}




