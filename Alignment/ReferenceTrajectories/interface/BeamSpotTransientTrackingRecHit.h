#ifndef BeamSpotTransientTrackingRecHit_H
#define BeamSpotTransientTrackingRecHit_H

/** \class BeamSpotTransientTrackingRecHit
 *
 * Transient tracking rec hit for the beam spot used in ReferenceTrajectory
 * to extend the track to the beam spot.
 *
 * Author     : Andreas Mussgiller
 * date       : 2010/08/30
 * last update: $Date: 2011/05/18 10:19:12 $
 * by         : $Author: mussgill $
 */

#include <cmath>

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
    :TransientTrackingRecHit(geom, AlignableBeamSpot::detId(), valid) {

    localPosition_ = det()->toLocal(GlobalPoint(beamSpot.x0(), beamSpot.y0(), beamSpot.z0()));
    localError_ = LocalError(std::pow(beamSpot.BeamWidthX()*cos(phi), 2) +
		  	     std::pow(beamSpot.BeamWidthY()*sin(phi), 2),
		             0.0, std::pow(beamSpot.sigmaZ(), 2));
  }
    
  virtual ~BeamSpotTransientTrackingRecHit() {}

  virtual LocalPoint localPosition() const { return localPosition_; }
  virtual LocalError localPositionError() const { return localError_; }

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

  LocalPoint localPosition_;
  LocalError localError_;

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

