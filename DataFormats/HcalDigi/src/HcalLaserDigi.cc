#include "DataFormats/HcalDigi/interface/HcalLaserDigi.h"

HcalLaserDigi::HcalLaserDigi() {
  for (int i=0; i<32; ++i) qadcraw_[i]=0;
}

void HcalLaserDigi::setQADC(const std::vector<uint16_t>& values) {
  for (size_t  i=0; i<values.size() && i<32; ++i)
    qadcraw_[i]=values[i];
}

void HcalLaserDigi::addTDCHit(int channel, int hittime) {
  uint32_t packed=(hittime&0xFFFFFF)|((channel&0xFF)<<24);
  tdcraw_.push_back(packed);
}

int HcalLaserDigi::hitChannel(size_t ihit) const {
  if (ihit<tdcraw_.size()) return tdcraw_[ihit]>>24;
  else return -1;
}

int HcalLaserDigi::hitRaw(size_t ihit) const {
  if (ihit<tdcraw_.size()) return tdcraw_[ihit]&0xFFFFFF;
  else return -1;
}

double HcalLaserDigi::hitNS(size_t ihit) const {
  return hitRaw(ihit)*0.8;
}



