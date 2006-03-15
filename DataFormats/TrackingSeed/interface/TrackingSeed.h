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
// $Author: stevew $
// $Date: 2006/02/07 18:07:58 $
// $Revision: 1.2 $
//

#include <vector>
#include "DataFormats/TrackerRecHit2D/interface/BaseSiStripRecHit2DLocalPos.h"
#include "DataFormats/Common/interface/Ref.h"

class TrackingSeed {

public:

  typedef std::vector<BaseSiStripRecHit2DLocalPos const*>::iterator iterator;
  typedef std::vector<BaseSiStripRecHit2DLocalPos const*>::const_iterator const_iterator;

  TrackingSeed() {}

  inline void addHit(BaseSiStripRecHit2DLocalPos const *input) { detHits_.push_back(input); }

  inline iterator begin() { return detHits_.begin();}
  inline iterator end() { return detHits_.end();}

  inline const_iterator begin() const { return detHits_.begin();}
  inline const_iterator end() const { return detHits_.end();}

  inline unsigned int size() const { return detHits_.size(); }

private:

  std::vector<BaseSiStripRecHit2DLocalPos const*> detHits_;

};

#endif // DATAFORMATS_TRACKINGSEED_H
