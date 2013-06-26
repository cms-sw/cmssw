#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"


EcalTPGSpike::EcalTPGSpike()
{ }

EcalTPGSpike::~EcalTPGSpike()
{ }

void EcalTPGSpike::setValue(const uint32_t & id, const uint16_t & val)
{
  map_[id] = val ;
}
