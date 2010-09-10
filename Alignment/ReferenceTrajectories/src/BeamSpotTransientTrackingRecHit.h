#ifndef BeamSpotTransientTrackingRecHit_H
#define BeamSpotTransientTrackingRecHit_H

/** \class BeamSpotTransientTrackingRecHit
 *
 * Transient tracking rec hit for the beam spot used in ReferenceTrajectory
 * to extend the track to the beam spot.
 *
 * Author     : Andreas Mussgiller
 * date       : 2010/08/30
 * last update: $Date: 2010/03/08 16:13:38 $
 * by         : $Author: mussgill $
 */

#include <iostream>

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

#include "Alignment/CommonAlignment/interface/AlignableBeamSpot.h"

#include "BeamSpotGeomDet.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class BeamSpotTransientTrackingRecHit: public TransientTrackingRecHit {
 public:

  typedef TrackingRecHit::Type Type;
  
  BeamSpotTransientTrackingRecHit(const reco::BeamSpot &beamSpot,
				  const BeamSpotGeomDet * geom,
				  double phi)
    :TransientTrackingRecHit(geom, AlignableBeamSpot::detId(), valid, 1.0, 1.0) {

    beamSpotGlobalPosition_ = GlobalPoint(beamSpot.x0(), beamSpot.y0(), beamSpot.z0());
    beamSpotLocalPosition_ = det()->toLocal(beamSpotGlobalPosition_);

    beamSpotLocalError_ = LocalError(sqrt(beamSpot.BeamWidthX()*cos(phi)*beamSpot.BeamWidthX()*cos(phi) +
					  beamSpot.BeamWidthY()*sin(phi)*beamSpot.BeamWidthY()*sin(phi)),
				     0.0, beamSpot.sigmaZ());
    beamSpotGlobalError_ = ErrorFrameTransformer().transform(beamSpotLocalError_, det()->surface());
    
    beamSpotErrorRPhi_ = beamSpotGlobalPosition_.perp()*sqrt(beamSpotGlobalError_.phierr(beamSpotGlobalPosition_)); 
    beamSpotErrorR_ = sqrt(beamSpotGlobalError_.rerr(beamSpotGlobalPosition_));
    beamSpotErrorZ_ = sqrt(beamSpotGlobalError_.czz());
  }
    
  virtual ~BeamSpotTransientTrackingRecHit() {}

  virtual GlobalPoint globalPosition() const { return beamSpotGlobalPosition_; }
  virtual GlobalError globalPositionError() const { return beamSpotGlobalError_; }

  virtual LocalPoint localPosition() const { return beamSpotLocalPosition_; }
  virtual LocalError localPositionError() const { return beamSpotLocalError_; }

  float errorGlobalR() const { return beamSpotErrorR_; }
  float errorGlobalZ() const { return beamSpotErrorZ_; }
  float errorGlobalRPhi() const { return beamSpotErrorRPhi_; }

  virtual AlgebraicVector parameters() const;
  virtual AlgebraicSymMatrix parametersError() const;
  virtual int dimension() const { return 1; }

  virtual const TrackingRecHit * hit() const { return 0; }

  virtual std::vector<const TrackingRecHit*> recHits() const {
    return std::vector<const TrackingRecHit*>();
  }
  virtual std::vector<TrackingRecHit*> recHits() {
    return std::vector<TrackingRecHit*>();
  }

  virtual const Surface * surface() const { return &(det()->surface()); }

  virtual AlgebraicMatrix projectionMatrix() const {
    if (!isInitialized) initialize();
    return theProjectionMatrix;
  }

 protected:

  GlobalPoint beamSpotGlobalPosition_;
  GlobalError beamSpotGlobalError_;  
  float beamSpotErrorR_, beamSpotErrorZ_, beamSpotErrorRPhi_;
  LocalPoint beamSpotLocalPosition_;
  LocalError beamSpotLocalError_;

 private:
  
  // should not have assignment operator (?)
  BeamSpotTransientTrackingRecHit & operator= (const BeamSpotTransientTrackingRecHit & t) {
     return *(this);
  }

  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
   virtual BeamSpotTransientTrackingRecHit * clone() const {
     return new BeamSpotTransientTrackingRecHit(*this);
   }
   
   static bool isInitialized;
   static AlgebraicMatrix theProjectionMatrix;
   void initialize() const;
};

#endif

