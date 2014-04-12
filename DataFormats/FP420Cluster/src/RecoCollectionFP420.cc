///////////////////////////////////////////////////////////////////////////////
// File: RecoCollectionFP420.cc
// Date: 11.2007
// Description: RecoCollectionFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "DataFormats/FP420Cluster/interface/RecoCollectionFP420.h"
#include <algorithm>
#include <iostream>
using namespace std;
//#define mydigidebug

void RecoCollectionFP420::put(RecoCollectionFP420::Range input, unsigned int stationID) {
  // put in RecHits of stationID

  // store size of vector before put
  RecoCollectionFP420::IndexRange inputRange;
  
  // put in RecoFP420s from input
  bool first = true;
  
  // fill input in temporary vector for sorting
  std::vector<RecoFP420> temporary;
  RecoCollectionFP420::ContainerIterator sort_begin = input.first;
  RecoCollectionFP420::ContainerIterator sort_end = input.second;
#ifdef mydigidebug
   std::cout <<"   !!!!!!!!!!!!!!!!    RecoCollectionFP420:: !!!!  put !!!!           start " << std::endl;
#endif
  for ( ;sort_begin != sort_end; ++sort_begin ) {
#ifdef mydigidebug
   std::cout <<"put: temporary.push_back " << std::endl;
#endif
    temporary.push_back(*sort_begin);
  }
  std::sort(temporary.begin(),temporary.end());

  // iterators over input
  RecoCollectionFP420::ContainerIterator begin = temporary.begin();
  RecoCollectionFP420::ContainerIterator end = temporary.end();
  for ( ;begin != end; ++begin ) {
    container_.push_back(*begin);
    if ( first ) {
          inputRange.first = container_.size()-1;
 //     inputRange.first = container_.size();
      first = false;
    }
  }
  
  // since we start from 0, then the last element will be size-1
  if(container_.size() != 0) {
    inputRange.second = container_.size()-1;
  }
  else {
    inputRange.first = container_.size();
    inputRange.second = container_.size();
  }

//  inputRange.second = container_.size()-1;
//  //inputRange.second = container_.size();

#ifdef mydigidebug
   std::cout <<"put: container_.size() = " << container_.size() << std::endl;
   std::cout <<"put:  inputRange.first = " << inputRange.first << std::endl;
   std::cout <<"put:  inputRange.second = " << inputRange.second << std::endl;
#endif
  
  // fill map
  map_[stationID] = inputRange;

}

const RecoCollectionFP420::Range RecoCollectionFP420::get(unsigned int stationID) const {
  // get RecHits of stationID

#ifdef mydigidebug
std::cout <<"RecoCollectionFP420::get:stationID= " << stationID << std::endl;
#endif
  RecoCollectionFP420::RegistryIterator returnIndex = map_.find(stationID);
  RecoCollectionFP420::IndexRange returnIndexRange = returnIndex->second;
#ifdef mydigidebug
   std::cout <<"RecoCollectionFP420::get1: returnIndexRange.first= " << returnIndexRange.first << std::endl;
   std::cout <<"RecoCollectionFP420::get1: returnIndexRange.second= " << returnIndexRange.second << std::endl;
#endif

  RecoCollectionFP420::Range returnRange;
  returnRange.first  = container_.begin()+returnIndexRange.first;
  if(returnIndexRange.second != 0 ) {
    returnRange.second = container_.begin()+returnIndexRange.second+1;
  }else{
    returnRange.second = container_.begin()+returnIndexRange.second;
  }
#ifdef mydigidebug
   std::cout <<"RecoCollectionFP420::get2: container_.size() = " << container_.size() << std::endl;
   std::cout <<"RecoCollectionFP420::get2: returnIndexRange.first= " << returnIndexRange.first << std::endl;
   std::cout <<"RecoCollectionFP420::get2: returnIndexRange.second= " << returnIndexRange.second << std::endl;
#endif
  return returnRange;
}



void RecoCollectionFP420::clear() {
container_.clear();
}
void RecoCollectionFP420::putclear(RecoCollectionFP420::Range input, unsigned int stationID) {

  RecoCollectionFP420::IndexRange inputRange;
  
  std::vector<RecoFP420> temporary;
  RecoCollectionFP420::ContainerIterator sort_begin = input.first;
  RecoCollectionFP420::ContainerIterator sort_end = input.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(*sort_begin);
  }
  std::sort(temporary.begin(),temporary.end());

  //	temporary.clear();
  RecoCollectionFP420::ContainerIterator begin = temporary.begin();
  RecoCollectionFP420::ContainerIterator end = temporary.end();
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


	  map_[stationID] = inputRange;
}






const std::vector<unsigned int> RecoCollectionFP420::stationIDs() const {
  // returns vector of stationIDs in map

#ifdef mydigidebug
std::cout <<"RecoCollectionFP420::stationIDs:start " << std::endl;
#endif
  RecoCollectionFP420::RegistryIterator begin = map_.begin();
  RecoCollectionFP420::RegistryIterator end   = map_.end();

  std::vector<unsigned int> output;

  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }

  return output;

}
