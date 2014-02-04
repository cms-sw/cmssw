#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidthsSet.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
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
  }
  else {
    cell = std::find(mItems.begin(),mItems.end(), target);
  }
  if (cell == mItems.end() || 
      ((fId.det()==DetId::Hcal && HcalDetId(cell->id) != HcalDetId(fId)) ||
       (fId.det()==DetId::Calo && fId.subdetId()==HcalZDCDetId::SubdetectorId && HcalZDCDetId(cell->id) != HcalZDCDetId(fId)) ||
       (fId.det()!=DetId::Hcal && (fId.det()==DetId::Calo && fId.subdetId()!=HcalZDCDetId::SubdetectorId) && (cell->id != fId))))
    throw cms::Exception ("Conditions not found") << "Unavailable HcalCalibrationWidths for cell " << HcalGenericDetId(fId);
  return cell->calib;
}

void HcalCalibrationWidthsSet::setCalibrationWidths(DetId fId, const HcalCalibrationWidths& ca) {
  Item target(fId);
  sorted_=false;
  std::vector<Item>::iterator cell=std::find(mItems.begin(),mItems.end(),target); //slow, but guaranteed
  if (cell==mItems.end()) 
    {
      target.calib=ca;
      mItems.push_back(target);
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
