#ifndef DATAFORMATS_SISTRIPRECHIT2DMATCHEDLOCALPOSCOLLECTION_H
#define DATAFORMATS_SISTRIPRECHIT2DMATCHEDLOCALPOSCOLLECTION_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPos.h"
#include "DataFormats/Common/interface/own_vector.h"
#include "FWCore/EDProduct/interface/Ref.h"
#include <vector>
#include <map>
#include <utility>

class SiStripRecHit2DMatchedLocalPosCollection {

 public:

  typedef own_vector<SiStripRecHit2DMatchedLocalPos,ClonePolicy<SiStripRecHit2DMatchedLocalPos> >::iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;

  SiStripRecHit2DMatchedLocalPosCollection() {}
  
  void put(Range input, unsigned int detID);
  Range get(unsigned int detID) const;
  const std::vector<unsigned int> detIDs() const;
  
 private:
  mutable own_vector<SiStripRecHit2DMatchedLocalPos, ClonePolicy<SiStripRecHit2DMatchedLocalPos> >  container_;
  mutable Registry map_;

};

#endif // 


