#ifndef FastSimulation_Tracking_FastTransientTrackingRecHit_H
#define FastSimulation_Tracking_FastTransientTrackingRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 

class FastTransientTrackingRecHit: public TransientTrackingRecHit{
public:
  typedef TrackingRecHit::Type Type;

  virtual ~FastTransientTrackingRecHit() {;}

  virtual AlgebraicVector parameters() const {return trackingRecHit_->parameters();}
  virtual AlgebraicSymMatrix parametersError() const {return trackingRecHit_->parametersError();}
  virtual AlgebraicMatrix projectionMatrix() const {return trackingRecHit_->projectionMatrix();}
  virtual int dimension() const {return trackingRecHit_->dimension();}

  virtual LocalPoint localPosition() const {return trackingRecHit_->localPosition();}
  virtual LocalError localPositionError() const {return trackingRecHit_->localPositionError();}

  virtual bool canImproveWithTrack() const {return false;}

  virtual const TrackingRecHit * hit() const {return (const TrackingRecHit*)trackingRecHit_;};
  
  inline TrackingRecHit * hitPtr() const {return trackingRecHit_;};

  virtual std::vector<const TrackingRecHit*> recHits() const {
    return ((const TrackingRecHit*)(trackingRecHit_))->recHits();
  }

  virtual std::vector<TrackingRecHit*> recHits() {
    return trackingRecHit_->recHits();
  }

  /// public constructors are faster !
  FastTransientTrackingRecHit(const GeomDet * geom, const TrackingRecHit& rh) :
    TransientTrackingRecHit(geom,rh) {
    trackingRecHit_ = rh.clone();
  }

  /// Clone, but then must delete !
  FastTransientTrackingRecHit(const GeomDet * geom, const TrackingRecHit* rh) :
    TransientTrackingRecHit(geom,*rh) { 
    trackingRecHit_ = rh->clone();
  }

  FastTransientTrackingRecHit( const FastTransientTrackingRecHit & other ) :
    TransientTrackingRecHit( other.det(),other) {
    trackingRecHit_ = other.hit()->clone();
  }

  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
   virtual FastTransientTrackingRecHit * clone() const {
     return new FastTransientTrackingRecHit(*this);
   }

private:

  TrackingRecHit * trackingRecHit_;

  
};

#endif


