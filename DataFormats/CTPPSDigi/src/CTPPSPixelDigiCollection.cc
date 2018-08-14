#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigiCollection.h"

#include <algorithm>

void CTPPSPixelDigiCollection::put(Range input, unsigned int detID) {
/// put in Digis of detID

/// store size of vector before put
  IndexRange inputRange;

/// fill input in temporary vector for sorting
  std::vector<CTPPSPixelDigi> temporary;

  auto sort_begin = input.first;
  auto sort_end = input.second;

  temporary.insert(std::end(temporary), sort_begin, sort_end);

  std::sort(temporary.begin(),temporary.end());

  inputRange.first=container_.size();
  container_.insert(std::end(container_), std::begin(temporary), std::end(temporary));
  inputRange.second = container_.size()-1;
  
/// fill map
  map_[detID] = inputRange;

}

const CTPPSPixelDigiCollection::Range CTPPSPixelDigiCollection::get(unsigned int detID) const {
/// get Digis of detID
  
  auto found = map_.find(detID);
  CTPPSPixelDigiCollection::IndexRange returnIndexRange{};
  if(found != map_.end()) {
    returnIndexRange = found->second;
  }

  CTPPSPixelDigiCollection::Range returnRange;
  returnRange.first  = container_.begin()+returnIndexRange.first;
  returnRange.second = container_.begin()+returnIndexRange.second+1;

  return returnRange;
}

const std::vector<unsigned int> CTPPSPixelDigiCollection::detIDs() const {
/// returns vector of detIDs in map

  auto begin = map_.begin();
  auto end   = map_.end();

  std::vector<unsigned int> output;
  output.reserve(12);

  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }

  return output;

}
