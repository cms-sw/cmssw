#ifndef DATAFORMATS_ROADSEARCHSEED_H
#define DATAFORMATS_ROADSEARCHSEED_H

//
// Package:         DataFormats/RoadSearchSeed
// Class:           RoadSearchSeed
// 
// Description:     seed holding non-persistent pointers to
//                  three hits of seed plus RoadSet
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Fri Jun 22 12:32:25 UTC 2007
//
// $Author: gutsche $
// $Date: 2006/08/29 22:15:40 $
// $Revision: 1.4 $
//

#include <vector>

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/RoadMapRecord/interface/Roads.h"

class RoadSearchSeed {
public:

  typedef std::vector<const TrackingRecHit*> HitVector;

  RoadSearchSeed() {}

  inline const Roads::RoadSeed*    getSeed() const                           { return seed_;             }
  inline void                      setSeed(const Roads::RoadSeed *input)     { seed_ = input;            }

  inline const Roads::RoadSet*     getSet() const                            { return set_;              }
  inline void                      setSet(const Roads::RoadSet *input)       { set_ = input;             }

  inline void                      addHit(const TrackingRecHit *input) { hits_.push_back(input);  }
  inline HitVector::const_iterator begin() const                       { return hits_.begin();    }
  inline HitVector::const_iterator end() const                         { return hits_.end();      }
  inline unsigned int              nHits() const                       { return hits_.size();     }

private:

  const Roads::RoadSeed                    *seed_;
  const Roads::RoadSet                     *set_;
  std::vector<const TrackingRecHit*>        hits_;

};

#endif // DATAFORMATS_ROADSEARCHSEED_H
