#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include <iostream>

void SiStripClusterCollection::put(SiStripClusterCollection::Range input, unsigned int detID) {
  // put in RecHits of detID

  // store size of vector before put
  SiStripClusterCollection::IndexRange inputRange;
  
  // put in SiStripClusters from input
  bool first = true;
  
  // fill input in temporary vector for sorting
  std::vector<SiStripCluster> temporary;
  SiStripClusterCollection::ContainerIterator sort_begin = input.first;
  SiStripClusterCollection::ContainerIterator sort_end = input.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(*sort_begin);
  }
  std::sort(temporary.begin(),temporary.end());

  // iterators over input
  SiStripClusterCollection::ContainerIterator begin = temporary.begin();
  SiStripClusterCollection::ContainerIterator end = temporary.end();
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

const SiStripClusterCollection::Range SiStripClusterCollection::get(unsigned int detID) const {
  // get RecHits of detID

  SiStripClusterCollection::RegistryIterator returnIndex = map_.find(detID);
  SiStripClusterCollection::IndexRange returnIndexRange = returnIndex->second;

  SiStripClusterCollection::Range returnRange;
  returnRange.first  = container_.begin()+returnIndexRange.first;
  returnRange.second = container_.begin()+returnIndexRange.second+1;

  return returnRange;
}

const std::vector<unsigned int> SiStripClusterCollection::detIDs() const {
  // returns vector of detIDs in map

  SiStripClusterCollection::RegistryIterator begin = map_.begin();
  SiStripClusterCollection::RegistryIterator end   = map_.end();

  std::vector<unsigned int> output;

  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }

  return output;

}
