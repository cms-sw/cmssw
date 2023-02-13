// -*- C++ -*-
//
// Package:    CalibTracker/SiStripHitResolution
// Class:      SiStripOverlapHit
//
/**\class SiStripOverlapHit SiStripOverlapHit.h CalibTracker/SiStripHitResolution/interface/SiStripOverlapHit.cc

 Description: A pair of hits on overlaping modules

 Implementation:
     Designed for CPE studies. Includes methods to compute residuals, etc.
*/
//
// Original Author:  Christophe Delaere
//         Created:  Fri, 20 Sep 2019 14:45:00 GMT
//
//
#ifndef CalibTracker_SiStripHitResolution_SiStripOverlapHit_H
#define CalibTracker_SiStripHitResolution_SiStripOverlapHit_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class SiStripOverlapHit {
public:
  using RecHitPointer = TrackingRecHit::RecHitPointer;
  using ConstRecHitPointer = TrackingRecHit::ConstRecHitPointer;

  // constructes an overlap from 2 hits and a track. Hits are internally sorted inside-out
  explicit SiStripOverlapHit(TrajectoryMeasurement const& measA, TrajectoryMeasurement const& measB);
  // destructor
  virtual ~SiStripOverlapHit(){};

  // access to indivitual hits and to the trajectory state
  inline ConstRecHitPointer const& hitA() const { return measA_.recHit(); }
  inline ConstRecHitPointer const& hitB() const { return measB_.recHit(); }
  inline ConstRecHitPointer const& hit(unsigned int hit) const { return hit ? hitB() : hitA(); }
  TrajectoryStateOnSurface const& trajectoryStateOnSurface(unsigned int hit = 0, bool updated = true) const;

  // utilities

  // track local angle
  double getTrackLocalAngle(unsigned int hit) const;  // 0: average, 1: hit A, 2: hit B
  // raw local distance between hit and strajectory state
  double offset(unsigned int hit) const;
  // distance between the two hits in the "trajectory frame"
  double shift() const;
  // absolute global distance between the hits (useful to debug pair finding)
  double distance(bool fromTrajectory = false) const;
  // global position (averaged over the two hits)
  GlobalPoint position() const;

private:
  TrajectoryMeasurement measA_;
  TrajectoryMeasurement measB_;
};

#endif
