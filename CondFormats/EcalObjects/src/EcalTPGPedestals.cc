#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"


EcalTPGPedestals::EcalTPGPedestals()
{ }

EcalTPGPedestals::~EcalTPGPedestals()
{ }

void EcalTPGPedestals::setValue(const uint32_t & id, const Item & value)
{
  map_[id] = value ;
}
