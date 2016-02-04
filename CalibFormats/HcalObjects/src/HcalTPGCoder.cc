#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"

std::vector<unsigned short> HcalTPGCoder::getLinearizationLUT(HcalDetId id) const {
  std::vector<unsigned short> lut(128);
  for (unsigned char i=0; i<128; ++i) lut[i]=adc2Linear(i,id);
  return lut;
}
