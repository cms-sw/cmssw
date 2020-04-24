#ifndef BeamSpotTransientTrackingRecHit_H
#define BeamSpotTransientTrackingRecHit_H

/** \class BeamSpotTransientTrackingRecHit
 *
 * Transient tracking rec hit for the beam spot used in ReferenceTrajectory
 * to extend the track to the beam spot.
 *
 * Author     : Andreas Mussgiller
 * date       : 2010/08/30
 * last update: $Date: 2012/02/04 15:02:59 $
 * by         : $Author: innocent $
 */

#include <cmath>

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

#include "Alignment/CommonAlignment/interface/AlignableBeamSpot.h"

#include "BeamSpotGeomDet.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"

class BeamSpotTransientTrackingRecHit final : public TValidTrackingRecHit {
 public:

  typedef TrackingRecHit::Type Type;
  
  BeamSpotTransientTrackingRecHit(const reco::BeamSpot &beamSpot,
				  const BeamSpotGeomDet * geom,
				  double phi)
    : TValidTrackingRecHit(*geom) {

    localPosition_ = det()->toLocal(GlobalPoint(beamSpot.x0(), beamSpot.y0(), beamSpot.z0()));
    localError_ = LocalError(std::pow(beamSpot.BeamWidthX()*cos(phi), 2) +
		  	     std::pow(beamSpot.BeamWidthY()*sin(phi), 2),
		             0.0, std::pow(beamSpot.sigmaZ(), 2));
  }
    
  ~BeamSpotTransientTrackingRecHit() override {}

  LocalPoint localPosition() const override { return localPosition_; }
  LocalError localPositionError() const override { return localError_; }

  AlgebraicVector parameters() const override;
  AlgebraicSymMatrix parametersError() const override;
  int dimension() const override { return 1; }

  const TrackingRecHit * hit() const override { return nullptr; }
  TrackingRecHit * cloneHit() const override { return nullptr;}


  std::vector<const TrackingRecHit*> recHits() const override {
    return std::vector<const TrackingRecHit*>();
  }
  std::vector<TrackingRecHit*> recHits() override {
    return std::vector<TrackingRecHit*>();
  }

  AlgebraicMatrix projectionMatrix() const override {
    return theProjectionMatrix;
  }

 protected:

  LocalPoint localPosition_;
  LocalError localError_;

 private:
  
  // should not have assignment operator (?)
  BeamSpotTransientTrackingRecHit & operator= (const BeamSpotTransientTrackingRecHit & t) {
     return *(this);
  }

  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
   BeamSpotTransientTrackingRecHit * clone() const override {
     return new BeamSpotTransientTrackingRecHit(*this);
   }
   
   static const AlgebraicMatrix theProjectionMatrix;
};

#endif

