///////////////////////////////////////////////////////////////////////////////
// File: DigiCollectionFP420.cc
// Date: 12.2006
// Description: DigiCollectionFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
//#include "SimRomanPot/SimFP420/interface/DigiCollectionFP420.h"
//#include "SimRomanPot/DataFormats/interface/DigiCollectionFP420.h"
#include "DataFormats/FP420Digi/interface/DigiCollectionFP420.h"

#include <algorithm>
//#include <iostream>
//#define mydigidebug

void DigiCollectionFP420::put(Range input, unsigned int detID) {
  // put in Digis of detID
  
  // store size of vector before put
  IndexRange inputRange;
  
  // put in HDigiFP420s from input
  bool first = true;
  
  // fill input in temporary vector for sorting
  std::vector<HDigiFP420> temporary;
  DigiCollectionFP420::ContainerIterator sort_begin = input.first;
  DigiCollectionFP420::ContainerIterator sort_end = input.second;
#ifdef mydigidebug
  std::cout <<"   !!!!!!!!!!!!!!!!    DigiCollectionFP420:: !!!!  put !!!!  start detID=" << detID << std::endl;
#endif
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(*sort_begin);
#ifdef mydigidebug
    std::cout <<"digi put: temporary.push_back " << std::endl;
#endif
  } // for
  std::sort(temporary.begin(),temporary.end());
  
  // iterators over input
  DigiCollectionFP420::ContainerIterator begin = temporary.begin();
  DigiCollectionFP420::ContainerIterator end = temporary.end();
  for ( ;begin != end; ++begin ) {
    container_.push_back(*begin);
    if ( first ) {
      inputRange.first = container_.size()-1;
#ifdef mydigidebug
      std::cout <<"digi put:first  container_.size() = " << container_.size() << std::endl;
      std::cout <<"digi put:first  inputRange.first = " << inputRange.first << std::endl;
#endif
      first = false;
    } // if
  } //for
  
  // since we start from 0, then the last element will be size-1
  if(container_.size() != 0) {
    inputRange.second = container_.size()-1;
  }
  else {
    inputRange.first = container_.size();
    inputRange.second = container_.size();
  }
#ifdef mydigidebug
  std::cout <<"digi put: container_.size() = " << container_.size() << std::endl;
  std::cout <<"digi put:  inputRange.first = " << inputRange.first << std::endl;
  std::cout <<"digi put:  inputRange.second = " << inputRange.second << std::endl;
#endif
  
  // fill map
  map_[detID] = inputRange;
  
}


void DigiCollectionFP420::clear() {
  container_.clear();
}
void DigiCollectionFP420::putclear(DigiCollectionFP420::Range input, unsigned int detID) {
  
  DigiCollectionFP420::IndexRange inputRange;
  
  std::vector<HDigiFP420> temporary;
  DigiCollectionFP420::ContainerIterator sort_begin = input.first;
  DigiCollectionFP420::ContainerIterator sort_end = input.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(*sort_begin);
  }
  std::sort(temporary.begin(),temporary.end());
  
  //	temporary.clear();
  DigiCollectionFP420::ContainerIterator begin = temporary.begin();
  DigiCollectionFP420::ContainerIterator end = temporary.end();
  for ( ;begin != end; ++begin ) {
    container_.push_back(*begin);
  }
  //container_.clear();
  inputRange.first = container_.size()-container_.size();
  inputRange.second = container_.size()-container_.size();
  
#ifdef mydigidebug
  std::cout <<"digi putclear: container_.size() = " << container_.size() << std::endl;
  std::cout <<"digi putclear:  inputRange.first = " << inputRange.first << std::endl;
  std::cout <<"digi putclear:  inputRange.second = " << inputRange.second << std::endl;
#endif
  
  
  map_[detID] = inputRange;
}






const DigiCollectionFP420::Range DigiCollectionFP420::get(unsigned int detID) const {
  // get Digis of detID
  
#ifdef mydigidebug
  std::cout <<"DigiCollectionFP420::get1:detID= " << detID << std::endl;
#endif
  auto found = map_.find(detID);
  if(found == map_.end()) {
    return DigiCollectionFP420::Range{container_.begin(),container_.begin()};
  }

  DigiCollectionFP420::IndexRange returnIndexRange = found->second;
  //
  DigiCollectionFP420::Range returnRange;
  returnRange.first  = container_.begin()+returnIndexRange.first;
  if(returnIndexRange.second != 0 ) {
    returnRange.second = container_.begin()+returnIndexRange.second+1;
  }else{
    returnRange.second = container_.begin()+returnIndexRange.second;
  }
  
#ifdef mydigidebug
  std::cout <<"digi get1: container_.size() = " << container_.size() << std::endl;
  std::cout <<"digi get1: returnIndexRange.first= " << returnIndexRange.first << std::endl;
  std::cout <<"digi get1: returnIndexRange.second= " << returnIndexRange.second << std::endl;
#endif
  return returnRange;
}


const DigiCollectionFP420::Range DigiCollectionFP420::get1(unsigned int detID) const {
  // get Digis of detID
  
#ifdef mydigidebug
  std::cout <<"DigiCollectionFP420::get :detID= " << detID << std::endl;
#endif
  DigiCollectionFP420::RegistryIterator returnIndex = map_.find(detID);
  DigiCollectionFP420::IndexRange returnIndexRange = returnIndex->second;
#ifdef mydigidebug
  std::cout <<"DigiCollectionFP420::get : returnIndexRange.first= " << returnIndexRange.first << std::endl;
  std::cout <<"DigiCollectionFP420::get : returnIndexRange.second= " << returnIndexRange.second << std::endl;
#endif
  
  DigiCollectionFP420::Range returnRange;
  returnRange.first  = container_.begin()+returnIndexRange.first;
  if(returnIndexRange.second != 0 ) {
    returnRange.second = container_.begin()+returnIndexRange.second+1;
  }else{
    returnRange.second = container_.begin()+returnIndexRange.second;
  }
#ifdef mydigidebug
  std::cout <<"DigiCollectionFP420::get : container_.size() = " << container_.size() << std::endl;
  std::cout <<"DigiCollectionFP420::get : returnIndexRange.first= " << returnIndexRange.first << std::endl;
  std::cout <<"DigiCollectionFP420::get : returnIndexRange.second= " << returnIndexRange.second << std::endl;
#endif
  return returnRange;
}







const std::vector<unsigned int> DigiCollectionFP420::detIDs() const {
  // returns vector of detIDs in map
  
  DigiCollectionFP420::RegistryIterator begin = map_.begin();
  DigiCollectionFP420::RegistryIterator end   = map_.end();
  
#ifdef mydigidebug
  std::cout <<"DigiCollectionFP420::detIDs:start " << std::endl;
#endif
  std::vector<unsigned int> output;
  
  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }
  
  return output;
  
}

// -----------------------------------------------------------------------------
// Appends HDigiFP420s to the vector of the given DetId.
void DigiCollectionFP420::add( unsigned int& det_id, 
			       std::vector<HDigiFP420>& digis ) {
  
#ifdef mydigidebug
  std::cout <<"DigiCollectionFP420::add:    det_id=    " << det_id << std::endl;
#endif
  digiMap_[det_id].reserve( digiMap_[det_id].size() + digis.size() );
  if ( digiMap_[det_id].empty() ) { 
    digiMap_[det_id] = digis;
  } else {
    copy( digis.begin(), digis.end(), back_inserter(digiMap_[det_id]) );
  }
}

// -----------------------------------------------------------------------------
// Returns (by reference) all Digis for a given DetId.
void DigiCollectionFP420::digis( unsigned int& det_id,
				 std::vector<HDigiFP420>& digis ) const {
#ifdef mydigidebug
  std::cout <<"DigiCollectionFP420::digis:det_id= " << det_id << std::endl;
#endif
  auto found = digiMap_.find( det_id );
  if ( found != digiMap_.end() ) { 
    digis = found->second;
  } else {
    digis = std::vector<HDigiFP420>();
  }
}

// -----------------------------------------------------------------------------
// Returns (by reference) vector of DetIds with Digis.
void DigiCollectionFP420::detIDs( std::vector<unsigned int>& det_ids ) const {
  det_ids.clear(); 
  det_ids.reserve( static_cast<unsigned int>(digiMap_.size()) );
#ifdef mydigidebug
  std::cout <<"DigiCollectionFP420::  detIDs:  digiMap    size= " << digiMap_.size() << std::endl;
#endif
  HDigiFP420Container::const_iterator iter;
  for (iter = digiMap_.begin(); iter != digiMap_.end(); iter++ ) {
    det_ids.push_back( iter->first );
  }
}
