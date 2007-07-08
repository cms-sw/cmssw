#ifndef DATAFORMATS_ROADSEARCHCLOUD_H
#define DATAFORMATS_ROADSEARCHCLOUD_H

//
// Package:         DataFormats/RoadSearchCloud
// Class:           RoadSearchCloud
// 
// Description:     Intermediate product of RoadSearch
//                  pattern recongnition. Holds refs to
//                  all RecHits in a Cloud following a Road.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/06/29 23:47:24 $
// $Revision: 1.5 $
//

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

class RoadSearchCloud {
public:

  typedef std::vector<const TrackingRecHit*> RecHitVector;

  RoadSearchCloud() {}
  RoadSearchCloud(RecHitVector rechits): recHits_(rechits) {}

  inline RoadSearchCloud* clone() const { return new RoadSearchCloud(recHits_); }
  inline void addHit(const TrackingRecHit* input) { recHits_.push_back(input); }
  inline unsigned int size() const { return recHits_.size(); }
  inline RecHitVector recHits() const { return recHits_; }
  inline RecHitVector::const_iterator begin_hits() const { return recHits_.begin(); }
  inline RecHitVector::const_iterator end_hits()   const { return recHits_.end();   }

private:

  RecHitVector recHits_;

};

#endif // DATAFORMATS_ROADSEARCHCLOUD_H
