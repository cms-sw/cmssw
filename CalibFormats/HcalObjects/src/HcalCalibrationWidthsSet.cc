#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidthsSet.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>
#include <iostream>


HcalCalibrationWidthsSet::HcalCalibrationWidthsSet() 
  : sorted_ (false) {}

const HcalCalibrationWidths& HcalCalibrationWidthsSet::getCalibrationWidths(const DetId fId) const {
  Item target(fId);
  std::vector<Item>::const_iterator cell;
  if (sorted_) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target);
  } else {
    cell = std::find(mItems.begin(),mItems.end(), target);
  }
  if ((cell == mItems.end()) || (!hcalEqualDetId(cell->id.rawId(),fId)))
    throw cms::Exception ("Conditions not found") << "Unavailable HcalCalibrationWidths for cell " << HcalGenericDetId(fId);
  return cell->calib;
}

void HcalCalibrationWidthsSet::setCalibrationWidths(DetId fId, const HcalCalibrationWidths& ca) {
  sorted_=false;
  std::vector<Item>::iterator cell=std::find(mItems.begin(),mItems.end(),Item(fId)); //slow, but guaranteed
  if (cell==mItems.end()) {
    mItems.push_back(Item(fId));
    mItems.at(mItems.size()-1).calib=ca;
    return;
  }
  cell->calib=ca;
}

void HcalCalibrationWidthsSet::sort () {
  if (!sorted_) {
    std::sort (mItems.begin(), mItems.end());
    sorted_ = true;
  }
}

void HcalCalibrationWidthsSet::clear() {
  mItems.clear();
}
