/** 
\class HcalGainWidths
\author Fedor Ratnikov (UMd)
POOL object to store pedestal values 4xCapId
$Author: ratnikov
$Date: 2005/08/02 01:31:24 $
$Revision: 1.2 $
*/

#include <iostream>

#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"

HcalGainWidths::HcalGainWidths() 
  : mSorted (false) {}

HcalGainWidths::~HcalGainWidths(){}

const float* HcalGainWidths::getValues (int fId) const {
  Item target;
  target.mId = fId;
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target);
  }
  else {
    std::cerr << "HcalPedestals::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = std::find (mItems.begin(), mItems.end(), target);
  }
  if (cell == mItems.end() || cell->mId != fId) return 0;
  return &(cell->mValue1);
}

float HcalGainWidths::getValue (int fId, int fCapId) const {
  const float* values;
  if (fCapId > 0 && fCapId <= 4) {
    values = getValues (fId);
    if (values) return values [fCapId];
  }
  else {
    std::cerr << "HcalPedestals::getValue-> capId " << fCapId << " is out of range [1..4]" << std::endl;
  }
  return -1;
}

bool HcalGainWidths::addValue (int fId, const float fValues [4]) {
  return addValue (fId, fValues [0], fValues [1], fValues [2], fValues [3]);
}

bool HcalGainWidths::addValue (int fId, float fValue1, float fValue2, float fValue3, float fValue4) {
  Item item;
  item.mId = fId;
  item.mValue1 = fValue1;
  item.mValue2 = fValue2;
  item.mValue3 = fValue3;
  item.mValue4 = fValue4;
  mItems.push_back (item);
  mSorted = false;
  return true;
}

std::vector<int> HcalGainWidths::getAllChannels () const {
  std::vector<int> result;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (item->mId);
  }
  return result;
}


void HcalGainWidths::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end());
    mSorted = true;
  }
}
