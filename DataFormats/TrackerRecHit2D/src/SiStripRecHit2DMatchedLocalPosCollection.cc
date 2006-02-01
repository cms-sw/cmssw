#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h"
#include "FWCore/EDProduct/interface/Ref.h"
#include <iostream>

void SiStripRecHit2DMatchedLocalPosCollection::put(SiStripRecHit2DMatchedLocalPosCollection::Range input, unsigned int detID) {
  // put in DetHits of detID

  // store size of vector before put
  SiStripRecHit2DMatchedLocalPosCollection::IndexRange inputRange;
  inputRange.first = container_.size();
  
  // put in SiStripRecHit2DMatchedLocalPoss from input
  
  // fill input in temporary vector for sorting
  Container temporary;
  SiStripRecHit2DMatchedLocalPosCollection::ContainerIterator sort_begin(input.first);
  SiStripRecHit2DMatchedLocalPosCollection::ContainerIterator sort_end(input.second);
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(new SiStripRecHit2DMatchedLocalPos(*sort_begin));
  }
  //  std::sort(temporary.begin(),temporary.end());

  // iterators over input
  SiStripRecHit2DMatchedLocalPosCollection::ContainerIterator begin = temporary.begin();
  SiStripRecHit2DMatchedLocalPosCollection::ContainerIterator end = temporary.end();
  for ( ;begin != end; ++begin ) {
    container_.push_back(new SiStripRecHit2DMatchedLocalPos(*begin));
  }
  inputRange.second = container_.size()-1;
  
  // fill map
  map_[detID] = inputRange;

}

SiStripRecHit2DMatchedLocalPosCollection::Range SiStripRecHit2DMatchedLocalPosCollection::get(unsigned int detID) const{
  // get DetHits of detID

  SiStripRecHit2DMatchedLocalPosCollection::RegistryIterator returnIndex = map_.find(detID);
  SiStripRecHit2DMatchedLocalPosCollection::IndexRange returnIndexRange = returnIndex->second;

  SiStripRecHit2DMatchedLocalPosCollection::ContainerIterator ibegin = container_.begin();
  SiStripRecHit2DMatchedLocalPosCollection::ContainerIterator iend = ibegin;
  ibegin += returnIndexRange.first;
  iend +=returnIndexRange.second+1;
  SiStripRecHit2DMatchedLocalPosCollection::Range returnRange(ibegin,iend);

  //  returnRange.first  = ibegin;
  //  returnRange.second = iend;
  // returnRange.first  = begin;
  // returnRange.second = begin;

  return returnRange;
}

const std::vector<unsigned int> SiStripRecHit2DMatchedLocalPosCollection::detIDs() const {
  // returns vector of detIDs in map

  SiStripRecHit2DMatchedLocalPosCollection::RegistryIterator begin = map_.begin();
  SiStripRecHit2DMatchedLocalPosCollection::RegistryIterator end   = map_.end();

  std::vector<unsigned int> output;
  
  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }

  return output;

}
