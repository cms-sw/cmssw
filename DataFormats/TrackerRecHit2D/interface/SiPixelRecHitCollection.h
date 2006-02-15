#ifndef DataFormats_SiPixelRecHitCollection_H
#define DataFormats_SiPixelRecHitCollection_H

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "FWCore/EDProduct/interface/Ref.h"
#include <vector>
#include <map>
#include <utility>

class SiPixelRecHitCollection {

 public:
  typedef edm::OwnVector<SiPixelRecHit, edm::ClonePolicy<SiPixelRecHit> >Container;
  typedef Container::iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;

  SiPixelRecHitCollection() {}
  
  void put(Range input, unsigned int detID);
  Range get(unsigned int detID) const;
  const std::vector<unsigned int> detIDs() const;
  
 private:
  mutable Container container_;
  mutable Registry map_;

};

#endif // 


