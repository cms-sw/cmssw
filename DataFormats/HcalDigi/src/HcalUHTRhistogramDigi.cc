#include "DataFormats/HcalDigi/interface/HcalUHTRhistogramDigiCollection.h"
#include <iomanip>

HcalUHTRhistogramDigi::HcalUHTRhistogramDigi(size_t index, const HcalUHTRhistogramDigiCollection& collection) : 
theCollection_(collection) {
  index_ = index;
}

uint32_t HcalUHTRhistogramDigi::get(int capid, int bin) const {
  return theCollection_.get(capid, bin, index_);
}

int HcalUHTRhistogramDigi::getSum(int bin) const {
  return theCollection_.getSum(bin, index_);
}


bool HcalUHTRhistogramDigi::separateCapIds() const { return theCollection_.separateCapIds(); }

bool HcalUHTRhistogramDigi::valid() const { return index_!=theCollection_.INVALID; }

const HcalDetId& HcalUHTRhistogramDigi::id() const { return theCollection_.id(index_); }

HcalUHTRhistogramDigiMutable::HcalUHTRhistogramDigiMutable(size_t index, HcalUHTRhistogramDigiCollection& collection) : HcalUHTRhistogramDigi(index, collection), theCollectionMutable_(collection) {
  
}
void HcalUHTRhistogramDigiMutable::fillBin(int capid, int bin, uint32_t val) {
  theCollectionMutable_.fillBin(capid, bin, val, index_);
}
