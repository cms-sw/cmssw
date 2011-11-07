#include "CondFormats/RPCObjects/interface/RPCNoiseStripsObject.h"
#include "FWCore/Utilities/interface/Exception.h"

std::vector<RPCNoiseStripsObject::NoiseStripsObjectItem>  const & 
RPCNoiseStripsObject::getCls() const
{
  return v_cls;
}




