#include "DataFormats/HcalRecHit/interface/HFPreRecHit.h"

std::pair<float, bool> HFPreRecHit::chargeAsymmetry(const float chargeThreshold) const {
  std::pair<float, bool> result(0.f, false);
  if (hasInfo_[0] && hasInfo_[1]) {
    const float q0 = hfQIE10Info_[0].charge();
    const float q1 = hfQIE10Info_[1].charge();
    const float qsum = q0 + q1;
    if (qsum > 0.f && qsum >= chargeThreshold) {
      result.first = (q1 - q0) / qsum;
      result.second = true;
    }
  }
  return result;
}

std::pair<float, bool> HFPreRecHit::energyAsymmetry(const float energyThreshold) const {
  std::pair<float, bool> result(0.f, false);
  if (hasInfo_[0] && hasInfo_[1]) {
    const float e0 = hfQIE10Info_[0].energy();
    const float e1 = hfQIE10Info_[1].energy();
    const float esum = e0 + e1;
    if (esum > 0.f && esum >= energyThreshold) {
      result.first = (e1 - e0) / esum;
      result.second = true;
    }
  }
  return result;
}
