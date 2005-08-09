#ifndef TRACKINGOBJECTS_STRIPDIGICOLLECTION_H
#define TRACKINGOBJECTS_STRIPDIGICOLLECTION_H

#include "DataFormats/SiStripDigi/interface/StripDigi.h"
#include <vector>
#include <map>
#include <utility>

class StripDigiCollection {

 public:

  typedef std::vector<StripDigi>::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;

  StripDigiCollection() {}
  
  void put(Range input, unsigned int detID);
  const Range get(unsigned int detID) const;
  const std::vector<unsigned int> detIDs() const;
  
 private:
  mutable std::vector<StripDigi> container_;
  mutable Registry map_;

};

#endif // TRACKINGOBJECTS_STRIPDIGICOLLECTION_H


