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
// $Date: 2006/01/15 00:56:20 $
// $Revision: 1.1 $
//

#include <vector>
#include "DataFormats/TrackingSeed/interface/TrackingSeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"

class RoadSearchCloud {
public:

  RoadSearchCloud() {}

  inline const std::vector<const SiStripRecHit2DLocalPos*> detHits() const { return detHits_;}
  inline void addHit(const SiStripRecHit2DLocalPos *input) { detHits_.push_back(input); }
  const TrackingSeed* seed() const { return seed_;}
  inline unsigned int size() const { return detHits_.size(); }

private:

  std::vector<const SiStripRecHit2DLocalPos*> detHits_;
  const TrackingSeed*                         seed_;

};

#endif // DATAFORMATS_ROADSEARCHCLOUD_H
