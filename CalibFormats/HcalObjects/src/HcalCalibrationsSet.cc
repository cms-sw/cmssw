#include "CalibFormats/HcalObjects/interface/HcalCalibrationsSet.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>
#include <iostream>


HcalCalibrationsSet::HcalCalibrationsSet() 
  : sorted_ (false) {}

const HcalCalibrations& HcalCalibrationsSet::getCalibrations(const DetId fId) const {
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
    throw cms::Exception ("Conditions not found") << "Unavailable HcalCalibrations for cell " << HcalGenericDetId(fId);
  return cell->calib;
}

void HcalCalibrationsSet::setCalibrations(DetId fId, const HcalCalibrations& ca) {
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
void HcalCalibrationsSet::sort () {
  if (!sorted_) {
    std::sort (mItems.begin(), mItems.end());
    sorted_ = true;
  }
}
void HcalCalibrationsSet::clear() {
  mItems.clear();
}
