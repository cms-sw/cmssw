///////////////////////////////////////////////////////////////////////////////
// File: ClusterCollectionFP420.cc
// Date: 12.2006
// Description: ClusterCollectionFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "DataFormats/FP420Cluster/interface/ClusterCollectionFP420.h"
#include <iostream>
#include <algorithm>
using namespace std;
//#define mydigidebug

void ClusterCollectionFP420::put(ClusterCollectionFP420::Range input, unsigned int detID) {
  // put in RecHits of detID

  // store size of vector before put
  ClusterCollectionFP420::IndexRange inputRange;
  
  // put in ClusterFP420s from input
  bool first = true;
  
  // fill input in temporary vector for sorting
  std::vector<ClusterFP420> temporary;
  ClusterCollectionFP420::ContainerIterator sort_begin = input.first;
  ClusterCollectionFP420::ContainerIterator sort_end = input.second;
#ifdef mydigidebug
   std::cout <<"   !!!!!!!!!!!!!!!!    ClusterCollectionFP420:: !!!!  put !!!!           start " << std::endl;
#endif
  for ( ;sort_begin != sort_end; ++sort_begin ) {
#ifdef mydigidebug
   std::cout <<"put: temporary.push_back " << std::endl;
#endif
    temporary.push_back(*sort_begin);
  }
  std::sort(temporary.begin(),temporary.end());

  // iterators over input
  ClusterCollectionFP420::ContainerIterator begin = temporary.begin();
  ClusterCollectionFP420::ContainerIterator end = temporary.end();
  for ( ;begin != end; ++begin ) {
    container_.push_back(*begin);
    if ( first ) {
          inputRange.first = container_.size()-1;
 //     inputRange.first = container_.size();
      first = false;
    }
  }

  // since we start from 0, then the last element will be size-1
  if(!container_.empty()) {
    inputRange.second = container_.size()-1;
  }
  else {
    inputRange.first = container_.size();
    inputRange.second = container_.size();
  }
  //inputRange.second = container_.size()-1;
  ////inputRange.second = container_.size();

#ifdef mydigidebug
   std::cout <<"put: container_.size() = " << container_.size() << std::endl;
   std::cout <<"put:  inputRange.first = " << inputRange.first << std::endl;
   std::cout <<"put:  inputRange.second = " << inputRange.second << std::endl;
#endif
  
  // fill map
  map_[detID] = inputRange;

}

const ClusterCollectionFP420::Range ClusterCollectionFP420::get(unsigned int detID) const {
  // get RecHits of detID

#ifdef mydigidebug
std::cout <<"ClusterCollectionFP420::get:detID= " << detID << std::endl;
#endif
  ClusterCollectionFP420::RegistryIterator returnIndex = map_.find(detID);
  ClusterCollectionFP420::IndexRange returnIndexRange = returnIndex->second;
#ifdef mydigidebug
   std::cout <<"ClusterCollectionFP420::get1: returnIndexRange.first= " << returnIndexRange.first << std::endl;
   std::cout <<"ClusterCollectionFP420::get1: returnIndexRange.second= " << returnIndexRange.second << std::endl;
#endif

  ClusterCollectionFP420::Range returnRange;
  returnRange.first  = container_.begin()+returnIndexRange.first;
  if(returnIndexRange.second != 0 ) {
    returnRange.second = container_.begin()+returnIndexRange.second+1;
  }else{
    returnRange.second = container_.begin()+returnIndexRange.second;
  }
#ifdef mydigidebug
   std::cout <<"ClusterCollectionFP420::get2: container_.size() = " << container_.size() << std::endl;
   std::cout <<"ClusterCollectionFP420::get2: returnIndexRange.first= " << returnIndexRange.first << std::endl;
   std::cout <<"ClusterCollectionFP420::get2: returnIndexRange.second= " << returnIndexRange.second << std::endl;
#endif
  return returnRange;
}



void ClusterCollectionFP420::clear() {
container_.clear();
}
void ClusterCollectionFP420::putclear(ClusterCollectionFP420::Range input, unsigned int detID) {

  ClusterCollectionFP420::IndexRange inputRange;
  
  std::vector<ClusterFP420> temporary;
  ClusterCollectionFP420::ContainerIterator sort_begin = input.first;
  ClusterCollectionFP420::ContainerIterator sort_end = input.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(*sort_begin);
  }
  std::sort(temporary.begin(),temporary.end());

  //	temporary.clear();
  ClusterCollectionFP420::ContainerIterator begin = temporary.begin();
  ClusterCollectionFP420::ContainerIterator end = temporary.end();
  for ( ;begin != end; ++begin ) {
    container_.push_back(*begin);
  }
  //container_.clear();
          inputRange.first = container_.size()-container_.size();
	  inputRange.second = container_.size()-container_.size();

#ifdef mydigidebug
   std::cout <<"putclear: container_.size() = " << container_.size() << std::endl;
   std::cout <<"putclear:  inputRange.first = " << inputRange.first << std::endl;
   std::cout <<"putclear:  inputRange.second = " << inputRange.second << std::endl;
#endif


	  map_[detID] = inputRange;
}






const std::vector<unsigned int> ClusterCollectionFP420::detIDs() const {
  // returns vector of detIDs in map

#ifdef mydigidebug
std::cout <<"ClusterCollectionFP420::detIDs:start " << std::endl;
#endif
  ClusterCollectionFP420::RegistryIterator begin = map_.begin();
  ClusterCollectionFP420::RegistryIterator end   = map_.end();

  std::vector<unsigned int> output;

  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }

  return output;

}
