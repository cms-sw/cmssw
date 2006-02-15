#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
//#include "FWCore/EDProduct/interface/Ref.h"
#include <iostream>

void SiPixelRecHitCollection::put(SiPixelRecHitCollection::Range input, unsigned int detID) {
  // put in DetHits of detID

  // store size of vector before put
  SiPixelRecHitCollection::IndexRange inputRange;
  inputRange.first = container_.size();
  
  // put in SiPixelRecHits from input
  
  // fill input in temporary vector for sorting
  Container temporary;
  SiPixelRecHitCollection::ContainerIterator sort_begin(input.first);
  SiPixelRecHitCollection::ContainerIterator sort_end(input.second);
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(new SiPixelRecHit(*sort_begin));
  }
  //  std::sort(temporary.begin(),temporary.end());

  // iterators over input
  SiPixelRecHitCollection::ContainerIterator begin = temporary.begin();
  SiPixelRecHitCollection::ContainerIterator end = temporary.end();
  for ( ;begin != end; ++begin ) {
    container_.push_back(new SiPixelRecHit(*begin));
  }
  inputRange.second = container_.size()-1;
  
  // fill map
  map_[detID] = inputRange;

}

SiPixelRecHitCollection::Range SiPixelRecHitCollection::get(unsigned int detID) const{
  // get DetHits of detID

  SiPixelRecHitCollection::RegistryIterator returnIndex = map_.find(detID);
  SiPixelRecHitCollection::IndexRange returnIndexRange = returnIndex->second;

  SiPixelRecHitCollection::ContainerIterator ibegin = container_.begin();
  SiPixelRecHitCollection::ContainerIterator iend = ibegin;
  ibegin += returnIndexRange.first;
  iend +=returnIndexRange.second+1;
  //StripRecHit2DLocalPosCollection::ContainerIterator begin = container_.begin();
  //SiPixelRecHitCollection::ContainerIterator end = container_.begin();
  SiPixelRecHitCollection::Range returnRange(ibegin,iend);

  //  returnRange.first  = ibegin;
  //  returnRange.second = iend;
  // returnRange.first  = begin;
  // returnRange.second = begin;

  return returnRange;
}

const std::vector<unsigned int> SiPixelRecHitCollection::detIDs() const {
  // returns vector of detIDs in map

  SiPixelRecHitCollection::RegistryIterator begin = map_.begin();
  SiPixelRecHitCollection::RegistryIterator end   = map_.end();

  std::vector<unsigned int> output;
  
  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }

  return output;

}
