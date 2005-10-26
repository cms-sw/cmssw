#ifndef TRACKINGOBJECTS_STRIPDIGICOLLECTION_H
#define TRACKINGOBJECTS_STRIPDIGICOLLECTION_H

#include "DataFormats/SiStripDigi/interface/StripDigi.h"
#include <utility>
#include <vector>
#include <map>

class StripDigiCollection {

 public:
  
  typedef std::vector<StripDigi>::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;

  /** Typedef for map of DetIds to their associated StripDigis. */
  typedef std::map< unsigned int, std::vector<StripDigi> > StripDigiContainer; // M.W.

  StripDigiCollection() {}
  
  void put(Range input, unsigned int detID);
  const Range get(unsigned int detID) const;
  const std::vector<unsigned int> detIDs() const;

  /** Appends StripDigis to the vector of the given DetId. */
  void add( unsigned int& det_id, std::vector<StripDigi>& digis ); // M.W
  /** Returns (by reference) all Digis for a given DetId. */
  void digis( unsigned int& det_id, std::vector<StripDigi>& digis ) const; // M.W
  /** Returns (by reference) vector of DetIds with Digis. */
  void detIDs( std::vector<unsigned int>& det_ids ) const; // M.W

 private:

  mutable std::vector<StripDigi> container_;
  mutable Registry map_;

  /** Map of DetIds to their associated StripDigis. */
  mutable StripDigiContainer digiMap_; // M.W


};

#endif // TRACKINGOBJECTS_STRIPDIGICOLLECTION_H


