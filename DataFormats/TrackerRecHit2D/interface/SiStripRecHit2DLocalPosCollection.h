#ifndef DATAFORMATS_SISTRIPRECHIT2DLOCALPOSCOLLECTION_H
#define DATAFORMATS_SISTRIPRECHIT2DLOCALPOSCOLLECTION_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "DataFormats/Common/interface/own_vector.h"
#include "FWCore/EDProduct/interface/Ref.h"
#include <vector>
#include <map>
#include <utility>

class SiStripRecHit2DLocalPosCollection {

 public:

  typedef own_vector<SiStripRecHit2DLocalPos,ClonePolicy<SiStripRecHit2DLocalPos> >::iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;

  SiStripRecHit2DLocalPosCollection() {}
  
  void put(Range input, unsigned int detID);
  Range get(unsigned int detID) const;
  const std::vector<unsigned int> detIDs() const;
  
 private:
  mutable own_vector<SiStripRecHit2DLocalPos, ClonePolicy<SiStripRecHit2DLocalPos> >  container_;
  mutable Registry map_;

};

#endif // 


