#ifndef DataFormats_SiPixelCluster_SiPixelClusterCollection_H
#define DataFormats_SiPixelCluster_SiPixelClusterCollection_H

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include <vector>
#include <map>
#include <utility>

class SiPixelClusterCollection {

 public:

  typedef std::vector<SiPixelCluster>::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;

  SiPixelClusterCollection() {}
  
  void put(Range input, unsigned int detID);
  const Range get(unsigned int detID) const;
  const std::vector<unsigned int> detIDs() const;
unsigned int size() const {return container_.size();}
  
 private:
  mutable std::vector<SiPixelCluster> container_;
  mutable Registry map_;

};

#endif


