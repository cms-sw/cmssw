/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store pedestal values 4xCapId
$Author: ratnikov
$Date: 2005/10/06 21:25:32 $
$Revision: 1.5 $
*/

#include <iostream>

#include "CondFormats/HcalObjects/interface/HcalQIEData.h"

namespace {
int index (int fCapId, int Range) {return fCapId*4+Range;}
}

HcalQIEData::HcalQIEData() 
  : mSorted (false) {}

HcalQIEData::~HcalQIEData(){}

const float* HcalQIEData::getOffsets (unsigned long fId) const {
  Item target;
  target.mId = fId;
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target);
  }
  else {
    std::cerr << "HcalQIEData::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = std::find (mItems.begin(), mItems.end(), target);
  }
  if (cell == mItems.end() || cell->mId != fId) return 0;
  return &(cell->mOffset00);
}

const float* HcalQIEData::getSlopes (unsigned long fId) const {
  Item target;
  target.mId = fId;
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target);
  }
  else {
    std::cerr << "HcalQIEData::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = std::find (mItems.begin(), mItems.end(), target);
  }
  if (cell == mItems.end() || cell->mId != fId) return 0;
  return &(cell->mSlope00);
}

  /// fill values [capid][range]
bool HcalQIEData::addValue (unsigned long fId, const float fOffsets [16], const float fSlopes [16]) {
  Item item;
  item.mId = fId;
  float* offset = &item.mOffset00;
  float* slope = &item.mSlope00;
  for (int i = 0; i < 16; i++) {
    offset [i] = fOffsets [i];
    slope [i] = fSlopes [i];
  }
  mItems.push_back (item);
  mSorted = false;
  return true;
}

std::vector<unsigned long> HcalQIEData::getAllChannels () const {
  std::vector<unsigned long> result;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (item->mId);
  }
  return result;
}


void HcalQIEData::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end());
    mSorted = true;
  }
}
