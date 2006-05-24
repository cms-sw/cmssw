#ifndef MuonTransientTrackingRecHit_H
#define MuonTransientTrackingRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"


class MuonTransientTrackingRecHit: public TransientTrackingRecHit{
public:

  /// constructor 
  MuonTransientTrackingRecHit(const GeomDet * geom, const RecSegment * rh) :
    TransientTrackingRecHit(geom), trackingRecHit_(rh) {
  }
 /// copy constructor 
  MuonTransientTrackingRecHit( const MuonTransientTrackingRecHit & other ) :
    TransientTrackingRecHit( other.det()),trackingRecHit_(other.hit())  {
  }

  virtual MuonTransientTrackingRecHit * clone() const {
    return new MuonTransientTrackingRecHit(*this);
  }

  virtual MuonTransientTrackingRecHit* clone( const TrajectoryStateOnSurface&) const {
    return clone();
  }

  virtual ~MuonTransientTrackingRecHit() {delete trackingRecHit_;}

  virtual AlgebraicVector parameters() const {return trackingRecHit_->parameters();}
  virtual AlgebraicSymMatrix parametersError() const {return trackingRecHit_->parametersError();}
  virtual DetId geographicalId() const {return trackingRecHit_->geographicalId();}
  virtual AlgebraicMatrix projectionMatrix() const {return trackingRecHit_->projectionMatrix();}
  virtual int dimension() const {return trackingRecHit_->dimension();}

  virtual LocalPoint localPosition() const {return trackingRecHit_->localPosition();}
  virtual LocalError localPositionError() const {return trackingRecHit_->localPositionError();}

  virtual const GeomDetUnit * detUnit() const;

  virtual bool canImproveWithTrack() const {return false;}

  virtual const RecSegment * hit() const {return trackingRecHit_;};
  
  virtual bool isValid() const{return trackingRecHit_->isValid();}

  virtual std::vector<const TrackingRecHit*> recHits() const {
    return trackingRecHit_->recHits();
  }
  virtual std::vector<TrackingRecHit*> recHits() {
    return ((TrackingRecHit *)(trackingRecHit_))->recHits();

  }

  virtual LocalVector localDirection() const {return trackingRecHit_->localDirection();}

  virtual GlobalVector globalDirection() const;

   /// Error on the local direction
  virtual LocalError localDirectionError() const {return trackingRecHit_->localDirectionError();}

   /// Error on the global direction
  virtual GlobalError globalDirectionError() const;
 
  virtual double chi2() const {return trackingRecHit_->chi2();}

  virtual int degreesOfFreedom() const {return trackingRecHit_->degreesOfFreedom();}

  /// assert if this rec hit is a DT rec hit 
  bool isDT() const;

  /// assert if this rec hit is a CSC rec hit 
  bool isCSC() const;

  //   /// assert if this rec hit is a RPC rec hit
  //   bool isRPC() const;
    
private:
  const RecSegment * trackingRecHit_;  
   
};
#endif

