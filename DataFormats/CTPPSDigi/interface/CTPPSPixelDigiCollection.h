#ifndef CTPPS_TRACKERDIGICOLLECTION_H
#define CTPPS_TRACKERDIGICOLLECTION_H

#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"
#include <vector>
#include <map>
#include <utility>

class CTPPSPixelDigiCollection {

 public:

  typedef std::vector<CTPPSPixelDigi>::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;

  CTPPSPixelDigiCollection() {}
  
  void put(Range input, unsigned int detID);
  const Range get(unsigned int detID) const;
  const std::vector<unsigned int> detIDs() const;
  
 private:
  std::vector<CTPPSPixelDigi> container_;
  Registry map_;

};

#endif 
