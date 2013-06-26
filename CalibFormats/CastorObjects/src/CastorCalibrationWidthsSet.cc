#include "CalibFormats/CastorObjects/interface/CastorCalibrationWidthsSet.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>
#include <iostream>


CastorCalibrationWidthsSet::CastorCalibrationWidthsSet() 
  : sorted_ (false) {}

const CastorCalibrationWidths& CastorCalibrationWidthsSet::getCalibrationWidths(const DetId fId) const {
  Item target(fId);
  std::vector<Item>::const_iterator cell;
  if (sorted_) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target);
  }
  else {
    cell = std::find(mItems.begin(),mItems.end(), target);
  }
  if (cell == mItems.end() || cell->id != fId) 
    throw cms::Exception ("Conditions not found") << "Unavailable CastorCalibrationWidths for cell " << HcalGenericDetId(fId);
  return cell->calib;
}

void CastorCalibrationWidthsSet::setCalibrationWidths(DetId fId, const CastorCalibrationWidths& ca) {
  sorted_=false;
  std::vector<Item>::iterator cell=std::find(mItems.begin(),mItems.end(),Item(fId)); //slow, but guaranteed
  if (cell==mItems.end()) 
    {
      mItems.push_back(Item(fId));
      mItems.at(mItems.size()-1).calib=ca;
      return;
    }
  cell->calib=ca;
}
void CastorCalibrationWidthsSet::sort () {
  if (!sorted_) {
    std::sort (mItems.begin(), mItems.end());
    sorted_ = true;
  }
}
void CastorCalibrationWidthsSet::clear() {
  mItems.clear();
}
