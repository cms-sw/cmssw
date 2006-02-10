#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "FWCore/EDProduct/interface/Ref.h"
#include <iostream>

void SiStripRecHit2DLocalPosCollection::put(SiStripRecHit2DLocalPosCollection::Range input, unsigned int detID) {
  // put in DetHits of detID

  // store size of vector before put
  IndexRange inputRange;
  inputRange.first = container_.size();
  
  // put in SiStripRecHit2DLocalPoss from input
  
  // fill input in temporary vector for sorting
  Container temporary;
  ContainerConstIterator sort_begin(input.first);
  ContainerConstIterator sort_end(input.second);
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(new SiStripRecHit2DLocalPos(*sort_begin));
  }
  //  std::sort(temporary.begin(),temporary.end());

  // iterators over input
  ContainerConstIterator begin = temporary.begin();
  ContainerConstIterator end = temporary.end();
  for ( ;begin != end; ++begin ) {
    container_.push_back(new SiStripRecHit2DLocalPos(*begin));
  }
  inputRange.second = container_.size()-1;
  
  // fill map
  map_[detID] = inputRange;

}

SiStripRecHit2DLocalPosCollection::Range SiStripRecHit2DLocalPosCollection::get(unsigned int detID) const{
  // get DetHits of detID

  RegistryIterator returnIndex = map_.find(detID);
  IndexRange returnIndexRange = returnIndex->second;

  ContainerConstIterator ibegin = container_.begin();
  ContainerConstIterator iend = ibegin;
  ibegin += returnIndexRange.first;
  iend +=returnIndexRange.second+1;
  //StripRecHit2DLocalPosCollection::ContainerIterator begin = container_.begin();
  //SiStripRecHit2DLocalPosCollection::ContainerIterator end = container_.begin();
  Range returnRange(ibegin,iend);

  //  returnRange.first  = ibegin;
  //  returnRange.second = iend;
  // returnRange.first  = begin;
  // returnRange.second = begin;

  return returnRange;
}

const std::vector<unsigned int> SiStripRecHit2DLocalPosCollection::detIDs() const {
  // returns vector of detIDs in map

  RegistryIterator begin = map_.begin();
  RegistryIterator end   = map_.end();

  std::vector<unsigned int> output;
  
  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }

  return output;

}
