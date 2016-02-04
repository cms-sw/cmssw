#include <iostream>

#include "RecoLocalCalo/CaloTowersCreator/interface/HcalMaterials.h"

HcalMaterials::HcalMaterials() {}

HcalMaterials::~HcalMaterials(){}

float HcalMaterials::getValue (DetId fId, float energy) {
  // a real function should be added
  float value = 0.;
  for(unsigned int iItem=0; iItem<mItems.size();iItem++){
    if(fId.rawId()==mItems[iItem].mmId()){
      value = mItems[iItem].getValue(energy);
      continue;
    }
  }
  return value;
}

void HcalMaterials::putValue (DetId fId, std::pair< std::vector <float>,std::vector <float> > fArray) {
  Item item (fId.rawId (), fArray);
  mItems.push_back (item);
}


