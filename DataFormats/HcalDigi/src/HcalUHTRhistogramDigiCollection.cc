#include "DataFormats/HcalDigi/interface/HcalUHTRhistogramDigiCollection.h"
#include <iomanip>
#include <iostream>
#include <algorithm>

const uint32_t HcalUHTRhistogramDigiCollection::get(int capid, int bin, size_t index) const {
  return bins_[(separateCapIds_*3+1)*binsPerHistogram_*index+capid*binsPerHistogram_+bin];  
}

const int HcalUHTRhistogramDigiCollection::getSum(int bin, size_t index) const {
  return bins_[(separateCapIds_*3+1)*binsPerHistogram_*index+0*binsPerHistogram_+bin]+  
         bins_[(separateCapIds_*3+1)*binsPerHistogram_*index+1*binsPerHistogram_+bin]+  
         bins_[(separateCapIds_*3+1)*binsPerHistogram_*index+2*binsPerHistogram_+bin]+  
         bins_[(separateCapIds_*3+1)*binsPerHistogram_*index+3*binsPerHistogram_+bin];  
}

void HcalUHTRhistogramDigiCollection::fillBin(int capid, int bin, uint32_t val, size_t index) {
  if ( index < ids_.size() )
    bins_[(separateCapIds_*3+1)*binsPerHistogram_*index+capid*binsPerHistogram_+bin] = val;
}
const HcalDetId& HcalUHTRhistogramDigiCollection::id(size_t index) const{
  return ids_[index];
}
const size_t HcalUHTRhistogramDigiCollection::find(HcalDetId id) const{
   std::cout << "Looking for: "<< id << std::endl;
   std::vector<HcalDetId>::const_iterator iter = std::find(ids_.begin(), ids_.end(), id);
   if (iter == ids_.end()) return INVALID;
   else return std::distance(ids_.begin() ,iter);
}

const HcalUHTRhistogramDigi HcalUHTRhistogramDigiCollection::at(size_t index) const {
  if (index >= ids_.size()) index = INVALID;
  return HcalUHTRhistogramDigi(index, *this );  
}
const HcalUHTRhistogramDigi HcalUHTRhistogramDigiCollection::operator[](size_t index) const {
  if (index >= ids_.size()) index = INVALID;
  return HcalUHTRhistogramDigi(index, *this );
}
HcalUHTRhistogramDigiMutable HcalUHTRhistogramDigiCollection::addHistogram(const HcalDetId& id) {
  size_t index = find(id);
  if ( index < ids_.size() ) {
    std::cout << "Found id: " << id << " at: " << index << ":" << ids_[index] << std::endl;
    return HcalUHTRhistogramDigiMutable(index, *this );
  }
  ids_.push_back(id);
  std::vector<uint32_t> res ((separateCapIds_*3+1)*binsPerHistogram_, 0);
  bins_.reserve( res.size() + bins_.size() );
  bins_.insert( bins_.end(), res.begin(), res.end() );
  std::cout << "Did not find id " << id << std::endl;
  std::cout << "#ids " << ids_.size() << std::endl;
  return HcalUHTRhistogramDigiMutable( ids_.size()-1, *this );
}
HcalUHTRhistogramDigiCollection::HcalUHTRhistogramDigiCollection() {}
HcalUHTRhistogramDigiCollection::HcalUHTRhistogramDigiCollection(int numBins, bool sepCapIds) {
  separateCapIds_ = sepCapIds;
  binsPerHistogram_ = numBins;
}
