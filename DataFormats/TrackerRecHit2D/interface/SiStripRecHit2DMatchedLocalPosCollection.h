#ifndef DATAFORMATS_SISTRIPRECHIT2DMATCHEDLOCALPOSCOLLECTION_H
#define DATAFORMATS_SISTRIPRECHIT2DMATCHEDLOCALPOSCOLLECTION_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPos.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "FWCore/EDProduct/interface/Ref.h"
#include <vector>
#include <map>
#include <utility>

class SiStripRecHit2DMatchedLocalPosCollection {

 public:
  typedef edm::OwnVector<SiStripRecHit2DMatchedLocalPos, edm::ClonePolicy<SiStripRecHit2DMatchedLocalPos> > Container;
  typedef Container::iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;

  SiStripRecHit2DMatchedLocalPosCollection() {}
  
  void put(Range input, unsigned int detID);
  Range get(unsigned int detID) const;
  const std::vector<unsigned int> detIDs() const;
  
 private:
  mutable Container container_;
  mutable Registry map_;

};

#endif // 


