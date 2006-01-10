#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "FWCore/EDProduct/interface/Ref.h"
#include <iostream>

void SiStripRecHit2DLocalPosCollection::put(SiStripRecHit2DLocalPosCollection::Range input, unsigned int detID) {
  // put in DetHits of detID

  // store size of vector before put
  SiStripRecHit2DLocalPosCollection::IndexRange inputRange;
  inputRange.first = container_.size();
  
  // put in SiStripRecHit2DLocalPoss from input
  
  // fill input in temporary vector for sorting
  own_vector<SiStripRecHit2DLocalPos, ClonePolicy<SiStripRecHit2DLocalPos> > temporary;
  SiStripRecHit2DLocalPosCollection::ContainerIterator sort_begin(input.first);
  SiStripRecHit2DLocalPosCollection::ContainerIterator sort_end(input.second);
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(new SiStripRecHit2DLocalPos(*sort_begin));
  }
  //  std::sort(temporary.begin(),temporary.end());

  // iterators over input
  SiStripRecHit2DLocalPosCollection::ContainerIterator begin = temporary.begin();
  SiStripRecHit2DLocalPosCollection::ContainerIterator end = temporary.end();
  for ( ;begin != end; ++begin ) {
    container_.push_back(new SiStripRecHit2DLocalPos(*begin));
  }
  inputRange.second = container_.size()-1;
  
  // fill map
  map_[detID] = inputRange;

}

SiStripRecHit2DLocalPosCollection::Range SiStripRecHit2DLocalPosCollection::get(unsigned int detID) const{
  // get DetHits of detID

  SiStripRecHit2DLocalPosCollection::RegistryIterator returnIndex = map_.find(detID);
  SiStripRecHit2DLocalPosCollection::IndexRange returnIndexRange = returnIndex->second;

  SiStripRecHit2DLocalPosCollection::ContainerIterator ibegin = container_.begin();
  SiStripRecHit2DLocalPosCollection::ContainerIterator iend = ibegin;
  ibegin += returnIndexRange.first;
  iend +=returnIndexRange.second+1;
  //StripRecHit2DLocalPosCollection::ContainerIterator begin = container_.begin();
  //SiStripRecHit2DLocalPosCollection::ContainerIterator end = container_.begin();
  SiStripRecHit2DLocalPosCollection::Range returnRange(ibegin,iend);

  //  returnRange.first  = ibegin;
  //  returnRange.second = iend;
  // returnRange.first  = begin;
  // returnRange.second = begin;

  return returnRange;
}

const std::vector<unsigned int> SiStripRecHit2DLocalPosCollection::detIDs() const {
  // returns vector of detIDs in map

  SiStripRecHit2DLocalPosCollection::RegistryIterator begin = map_.begin();
  SiStripRecHit2DLocalPosCollection::RegistryIterator end   = map_.end();

  std::vector<unsigned int> output;
  
  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }

  return output;

}
