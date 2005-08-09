#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include <iostream>

void StripDigiCollection::put(Range input, unsigned int detID) {
  // put in Digis of detID

  // store size of vector before put
  IndexRange inputRange;

  // put in StripDigis from input
  bool first = true;

  // fill input in temporary vector for sorting
  std::vector<StripDigi> temporary;
  StripDigiCollection::ContainerIterator sort_begin = input.first;
  StripDigiCollection::ContainerIterator sort_end = input.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(*sort_begin);
  }
  std::sort(temporary.begin(),temporary.end());

  // iterators over input
  StripDigiCollection::ContainerIterator begin = temporary.begin();
  StripDigiCollection::ContainerIterator end = temporary.end();
  for ( ;begin != end; ++begin ) {
    container_.push_back(*begin);
    if ( first ) {
      inputRange.first = container_.size()-1;
      first = false;
    }
  }
  inputRange.second = container_.size()-1;
  
  // fill map
  map_[detID] = inputRange;

}

const StripDigiCollection::Range StripDigiCollection::get(unsigned int detID) const {
  // get Digis of detID

  StripDigiCollection::IndexRange returnIndexRange = map_[detID];

  StripDigiCollection::Range returnRange;
  returnRange.first  = container_.begin()+returnIndexRange.first;
  returnRange.second = container_.begin()+returnIndexRange.second+1;

  return returnRange;
}

const std::vector<unsigned int> StripDigiCollection::detIDs() const {
  // returns vector of detIDs in map

  StripDigiCollection::RegistryIterator begin = map_.begin();
  StripDigiCollection::RegistryIterator end   = map_.end();

  std::vector<unsigned int> output;

  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }

  return output;

}
