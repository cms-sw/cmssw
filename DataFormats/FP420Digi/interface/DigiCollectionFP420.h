#ifndef DataFormats_DigiCollectionFP420_h
#define DataFormats_DigiCollectionFP420_h

//#include "SimRomanPot/SimFP420/interface/HDigiFP420.h"
//#include "SimRomanPot/DataFormats/interface/HDigiFP420.h"

#include "DataFormats/FP420Digi/interface/HDigiFP420.h"

#include <utility>
#include <vector>
#include <map>
#include <iostream>

class DigiCollectionFP420 {

 public:
  
  typedef std::vector<HDigiFP420>::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;
  typedef std::map< unsigned int, std::vector<HDigiFP420> > HDigiFP420Container; 

  DigiCollectionFP420() {}
  
  void put(Range input, unsigned int detID);
  const Range get(unsigned int detID) const;
  const Range get1(unsigned int detID) const;
  const std::vector<unsigned int> detIDs() const;

  void add( unsigned int& det_id, std::vector<HDigiFP420>& digis ); 
  void digis( unsigned int& det_id, std::vector<HDigiFP420>& digis ) const; 
  void detIDs( std::vector<unsigned int>& det_ids ) const; 

  void putclear(Range input, unsigned int detID);
  void clear();
 private:

  std::vector<HDigiFP420> container_;
  Registry map_;

  HDigiFP420Container digiMap_; 


};

#endif 


