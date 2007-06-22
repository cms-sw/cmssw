#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"


EcalTPGLinearizationConst::EcalTPGLinearizationConst()
{ }

EcalTPGLinearizationConst::~EcalTPGLinearizationConst()
{ }

void EcalTPGLinearizationConst::setValue(const uint32_t & id, const Item & value)
{
  map_[id] = value ;
}
