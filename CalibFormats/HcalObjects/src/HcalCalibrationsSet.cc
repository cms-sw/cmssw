#include "CalibFormats/HcalObjects/interface/HcalCalibrationsSet.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>
#include <iostream>
#include <utility>

HcalCalibrationsSet::HcalCalibrationsSet() 
{}

const HcalCalibrations& HcalCalibrationsSet::getCalibrations(const DetId fId) const {
  DetId fId2(hcalTransformedId(fId));
  auto cell = mItems.find(fId2);
  if ((cell == mItems.end()) || (!hcalEqualDetId(cell->first,fId2)))
    throw cms::Exception ("Conditions not found") << "Unavailable HcalCalibrations for cell " << HcalGenericDetId(fId);
  return cell->second.calib;
}

void HcalCalibrationsSet::setCalibrations(DetId fId, const HcalCalibrations& ca) {
  DetId fId2(hcalTransformedId(fId));
  auto cell = mItems.find(fId2);
  if (cell==mItems.end()) {
    auto result = mItems.emplace(fId2,fId2);
    result.first->second.calib=ca;
    return;
  }
  cell->second.calib=ca;
}

void HcalCalibrationsSet::clear() {
  mItems.clear();
}

std::vector<DetId> HcalCalibrationsSet::getAllChannels() const {
  std::vector<DetId> channels;
  channels.reserve(mItems.size());
  for(const auto& tmp : mItems){
    channels.push_back(tmp.second.id);
  }
  return channels;
}
