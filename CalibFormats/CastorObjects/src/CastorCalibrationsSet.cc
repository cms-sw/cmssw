#include "CalibFormats/CastorObjects/interface/CastorCalibrationsSet.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>


CastorCalibrationsSet::CastorCalibrationsSet() 
  : sorted_ (false) {}

const CastorCalibrations& CastorCalibrationsSet::getCalibrations(const DetId fId) const {
  Item target(fId);
  std::vector<Item>::const_iterator cell;
  if (sorted_) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target);
  }
  else {
    cell = std::find(mItems.begin(),mItems.end(), target);
  }
  if (cell == mItems.end() || cell->id != fId) 
    throw cms::Exception ("Conditions not found") << "Unavailable CastorCalibrations for cell " << HcalGenericDetId(fId);
  return cell->calib;
}

void CastorCalibrationsSet::setCalibrations(DetId fId, const CastorCalibrations& ca) {
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
void CastorCalibrationsSet::sort () {
  if (!sorted_) {
    std::sort (mItems.begin(), mItems.end());
    sorted_ = true;
  }
}
void CastorCalibrationsSet::clear() {
  mItems.clear();
}
