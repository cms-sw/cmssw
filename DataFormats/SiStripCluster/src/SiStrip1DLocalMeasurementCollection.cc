#include "DataFormats/SiStripCluster/interface/SiStrip1DLocalMeasurementCollection.h"
#include <iostream>

void SiStrip1DLocalMeasurementCollection::put(SiStrip1DLocalMeasurementCollection::Range input, unsigned int detID) {
  // put in RecHits of detID

  // store size of vector before put
  SiStrip1DLocalMeasurementCollection::IndexRange inputRange;
  
  // put in SiStrip1DLocalMeasurements from input
  bool first = true;
  
  // fill input in temporary vector for sorting
  std::vector<SiStrip1DLocalMeasurement> temporary;
  SiStrip1DLocalMeasurementCollection::ContainerIterator sort_begin = input.first;
  SiStrip1DLocalMeasurementCollection::ContainerIterator sort_end = input.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(*sort_begin);
  }
  std::sort(temporary.begin(),temporary.end());

  // iterators over input
  SiStrip1DLocalMeasurementCollection::ContainerIterator begin = temporary.begin();
  SiStrip1DLocalMeasurementCollection::ContainerIterator end = temporary.end();
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

const SiStrip1DLocalMeasurementCollection::Range SiStrip1DLocalMeasurementCollection::get(unsigned int detID) const {
  // get RecHits of detID

  SiStrip1DLocalMeasurementCollection::RegistryIterator returnIndex = map_.find(detID);
  SiStrip1DLocalMeasurementCollection::IndexRange returnIndexRange = returnIndex->second;

  SiStrip1DLocalMeasurementCollection::Range returnRange;
  returnRange.first  = container_.begin()+returnIndexRange.first;
  returnRange.second = container_.begin()+returnIndexRange.second+1;

  return returnRange;
}

const std::vector<unsigned int> SiStrip1DLocalMeasurementCollection::detIDs() const {
  // returns vector of detIDs in map

  SiStrip1DLocalMeasurementCollection::RegistryIterator begin = map_.begin();
  SiStrip1DLocalMeasurementCollection::RegistryIterator end   = map_.end();

  std::vector<unsigned int> output;

  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }

  return output;

}
