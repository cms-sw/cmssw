#include "DataFormats/HcalDigi/interface/HcalUHTRhistogramDigiCollection.h"
#include <iomanip>
#include <algorithm>

uint32_t HcalUHTRhistogramDigiCollection::get(int capid, int bin, int index) const {
  return bins_[(separateCapIds_*3+1)*binsPerHistogram_*index+capid*binsPerHistogram_+bin];  
}

int HcalUHTRhistogramDigiCollection::getSum(int bin, int index) const {
  return bins_[(separateCapIds_*3+1)*binsPerHistogram_*index+0*binsPerHistogram_+bin]+  
         bins_[(separateCapIds_*3+1)*binsPerHistogram_*index+1*binsPerHistogram_+bin]+  
         bins_[(separateCapIds_*3+1)*binsPerHistogram_*index+2*binsPerHistogram_+bin]+  
         bins_[(separateCapIds_*3+1)*binsPerHistogram_*index+3*binsPerHistogram_+bin];  
}

void HcalUHTRhistogramDigiCollection::fillBin(int capid, int bin, uint32_t val, HcalDetId id) {
  if ( abs(index(id)) < ids_.size() )
    bins_[(separateCapIds_*3+1)*binsPerHistogram_*index(id)+capid*binsPerHistogram_+bin] = val;
  else {
    ids_.push_back(id);
    std::vector<uint32_t> res ((separateCapIds_*3+1)*binsPerHistogram_, 0);
    bins_.reserve( res.size()+bins_.size() );
    bins_.insert( bins_.end(), res.begin(), res.end() );
    fillBin(capid, bin, val, id);
  }
}
const HcalDetId& HcalUHTRhistogramDigiCollection::id(int index) {
  return ids_[index];
}
int HcalUHTRhistogramDigiCollection::index(HcalDetId id) {
  return *(std::find(ids_.begin(), ids_.end(), id));
}
