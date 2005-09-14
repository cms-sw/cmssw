#ifndef DATAFORMATS_SISTRIP1DLOCALMEASUREMENTCOLLECTION_H
#define DATAFORMATS_SISTRIP1DLOCALMEASUREMENTCOLLECTION_H

#include "DataFormats/SiStripCluster/interface/SiStrip1DLocalMeasurement.h"
#include <vector>
#include <map>
#include <utility>

class SiStrip1DLocalMeasurementCollection {

 public:

  typedef std::vector<SiStrip1DLocalMeasurement>::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;

  SiStrip1DLocalMeasurementCollection() {}
  
  void put(Range input, unsigned int detID);
  const Range get(unsigned int detID) const;
  const std::vector<unsigned int> detIDs() const;
  
 private:
  mutable std::vector<SiStrip1DLocalMeasurement> container_;
  mutable Registry map_;

};

#endif // 


