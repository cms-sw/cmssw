#ifndef DATAFORMATS_TRACKINGSEED_H
#define DATAFORMATS_TRACKINGSEED_H

//
// Package:         DataFormats/TrackingSeed
// Class:           TrackingSeed
// 
// Description:     TrackingSeed represents the 
//                  initial trajectory for 
//                  track reconstruction
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/01/14 22:00:00 $
// $Revision: 1.1 $
//

#include <vector>
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "FWCore/EDProduct/interface/Ref.h"

class TrackingSeed {

public:

  typedef std::vector<SiStripRecHit2DLocalPos const*>::iterator iterator;
  typedef std::vector<SiStripRecHit2DLocalPos const*>::const_iterator const_iterator;

  TrackingSeed() {}

  inline void addHit(SiStripRecHit2DLocalPos const *input) { detHits_.push_back(input); }

  inline iterator begin() { return detHits_.begin();}
  inline iterator end() { return detHits_.end();}

  inline const_iterator begin() const { return detHits_.begin();}
  inline const_iterator end() const { return detHits_.end();}

  inline unsigned int size() const { return detHits_.size(); }

private:

  std::vector<SiStripRecHit2DLocalPos const*> detHits_;

};

#endif // DATAFORMATS_TRACKINGSEED_H
