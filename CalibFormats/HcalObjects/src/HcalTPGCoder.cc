#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"

std::vector<unsigned short> HcalTPGCoder::getLinearizationLUT(HcalDetId id) const {
  std::vector<unsigned short> lut(256);
  for (uint16_t i=0; i<=255; ++i) lut[i]=adc2Linear(i,id);
  return lut;
}
