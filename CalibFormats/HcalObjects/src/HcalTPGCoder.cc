#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

#include <vector>

std::vector<unsigned short> HcalTPGCoder::getLinearizationLUT(const HcalDetId& id) const {
  std::vector<unsigned short> lut(128);
  for (unsigned char i = 0; i < 128; ++i)
    lut[i] = adc2Linear(i, id);
  return lut;
}

std::vector<unsigned short> HcalTPGCoder::getLinearizationLUT(const HcalZDCDetId& id, bool ootpu_lut) const {
  std::vector<unsigned short> lut(128);
  for (unsigned char i = 0; i < 128; ++i)
    lut[i] = adc2Linear(i, id);
  return lut;
}
