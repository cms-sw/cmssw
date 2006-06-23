#include "DataFormats/SiPixelCluster/interface/SiPixelClusterCollection.h"
#include <iostream>

void 
SiPixelClusterCollection::put(SiPixelClusterCollection::Range input, 
			      unsigned int detID) 
{

  if (input.first == input.second) return ;
  // put in Clusters for a given detID

  // store size of vector before put
  SiPixelClusterCollection::IndexRange inputRange;
  
  // put in SiPixelClusters from input
  bool first = true;
  
  // fill input in temporary vector for sorting
  std::vector<SiPixelCluster> temporary;
  SiPixelClusterCollection::ContainerIterator sort_begin = input.first;
  SiPixelClusterCollection::ContainerIterator sort_end = input.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(*sort_begin);
  }
  std::sort(temporary.begin(),temporary.end());

  // iterators over input
  SiPixelClusterCollection::ContainerIterator begin = temporary.begin();
  SiPixelClusterCollection::ContainerIterator end = temporary.end();
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

const SiPixelClusterCollection::Range 
SiPixelClusterCollection::get(unsigned int detID) const 
{
  // get RecHits of detID

  SiPixelClusterCollection::RegistryIterator returnIndex = map_.find(detID);
  if (returnIndex== map_.end()){
    return Range(container_.end(),container_.end());
  }
  SiPixelClusterCollection::IndexRange returnIndexRange = returnIndex->second;

  SiPixelClusterCollection::Range returnRange;
  returnRange.first  = container_.begin()+returnIndexRange.first;
  returnRange.second = container_.begin()+returnIndexRange.second+1;

  return returnRange;
}


const std::vector<unsigned int> 
SiPixelClusterCollection::detIDs() const 
{
  // returns vector of detIDs in map

  SiPixelClusterCollection::RegistryIterator begin = map_.begin();
  SiPixelClusterCollection::RegistryIterator end   = map_.end();

  std::vector<unsigned int> output;

  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }

  return output;

}
