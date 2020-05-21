#include <iostream>

#include "RecoLocalCalo/CaloTowersCreator/interface/HcalMaterials.h"

HcalMaterials::HcalMaterials() {}

HcalMaterials::~HcalMaterials() {}

float HcalMaterials::getValue(DetId fId, float energy) {
  // a real function should be added
  float value = 0.;
  for (auto& mItem : mItems) {
    if (fId.rawId() == mItem.mmId()) {
      value = mItem.getValue(energy);
      continue;
    }
  }
  return value;
}

void HcalMaterials::putValue(DetId fId, const std::pair<std::vector<float>, std::vector<float> >& fArray) {
  Item item(fId.rawId(), fArray);
  mItems.push_back(item);
}
