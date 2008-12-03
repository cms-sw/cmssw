#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"


EcalTPGTowerStatus::EcalTPGTowerStatus()
{ }

EcalTPGTowerStatus::~EcalTPGTowerStatus()
{ }

void EcalTPGTowerStatus::setValue(const uint32_t & id, const uint16_t & val)
{
  map_[id] = val ;
}
