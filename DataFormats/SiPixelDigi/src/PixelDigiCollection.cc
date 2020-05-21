#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include <iostream>
#include <algorithm>

void PixelDigiCollection::put(Range input, unsigned int detID) {
  // put in Digis of detID

  // store size of vector before put
  IndexRange inputRange;

  // put in PixelDigis from input
  bool first = true;

  // fill input in temporary vector for sorting
  std::vector<PixelDigi> temporary;
  auto sort_begin = input.first;
  auto sort_end = input.second;
  for (; sort_begin != sort_end; ++sort_begin) {
    temporary.push_back(*sort_begin);
  }
  std::sort(temporary.begin(), temporary.end());

  // iterators over input
  auto begin = temporary.begin();
  auto end = temporary.end();
  for (; begin != end; ++begin) {
    container_.push_back(*begin);
    if (first) {
      inputRange.first = container_.size() - 1;
      first = false;
    }
  }
  inputRange.second = container_.size() - 1;

  // fill map
  map_[detID] = inputRange;
}

const PixelDigiCollection::Range PixelDigiCollection::get(unsigned int detID) const {
  // get Digis of detID

  auto found = map_.find(detID);
  PixelDigiCollection::IndexRange returnIndexRange{};
  if (found != map_.end()) {
    returnIndexRange = found->second;
  }

  PixelDigiCollection::Range returnRange;
  returnRange.first = container_.begin() + returnIndexRange.first;
  returnRange.second = container_.begin() + returnIndexRange.second + 1;

  return returnRange;
}

const std::vector<unsigned int> PixelDigiCollection::detIDs() const {
  // returns vector of detIDs in map

  auto begin = map_.begin();
  auto end = map_.end();

  std::vector<unsigned int> output;

  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }

  return output;
}
