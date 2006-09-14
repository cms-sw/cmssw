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
// $Date: 2006/03/28 22:43:33 $
// $Revision: 1.3 $
//

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

class RoadSearchCloud {
public:

  typedef edm::OwnVector<TrackingRecHit,edm::ClonePolicy<TrackingRecHit> > RecHitOwnVector;
  typedef edm::Ref<TrajectorySeedCollection, TrajectorySeed> SeedRef;
  typedef edm::RefVector<TrajectorySeedCollection, TrajectorySeed> SeedRefs;

  RoadSearchCloud() {}
  RoadSearchCloud(RecHitOwnVector rechits, SeedRefs seedrefs): recHits_(rechits), seeds_(seedrefs) {}

  inline RoadSearchCloud* clone() const { return new RoadSearchCloud(recHits_,seeds_); }
  inline RecHitOwnVector recHits() const { return recHits_;}
  inline void addHit(TrackingRecHit* input) { recHits_.push_back(input); }
  inline void addSeed(SeedRef input) { seeds_.push_back(input); }
  inline SeedRefs seeds() const { return seeds_;}
  inline unsigned int size() const { return recHits_.size(); }
  inline RecHitOwnVector::const_iterator begin_hits() const { return recHits_.begin(); }
  inline RecHitOwnVector::const_iterator end_hits()   const { return recHits_.end();   }
  inline SeedRefs::const_iterator begin_seeds()       const { return seeds_.begin();   }
  inline SeedRefs::const_iterator end_seeds()         const { return seeds_.end();     }


private:

  RecHitOwnVector recHits_;
  SeedRefs     seeds_;

};

#endif // DATAFORMATS_ROADSEARCHCLOUD_H
