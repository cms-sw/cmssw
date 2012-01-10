#ifndef BaseTrackerRecHit_H
#define BaseTrackerRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

namespace trackerHitRTTI {
  // tracking hit can be : single (si1D, si2D, pix), projected, matched or multi
  enum RTTI { undef=0, single=1, proj=2, match=3, multi=4};
  RTTI rtti(TrackingRecHit const & hit) const { return RTTI(hit.getRTTI());}
  bool isUndef(TrackingRecHit const & hit) const { return rtti(hit)==undef;}
  bool isSingle(TrackingRecHit const & hit) const { return rtti(hit)==single;}
  bool isProjected(TrackingRecHit const & hit) const { return rtti(hit)==proj;}
  bool isMatched(TrackingRecHit const & hit) const { return rtti(hit)==match;}
  bool isMulti(TrackingRecHit const & hit) const { return rtti(hit)==multi;}


}

class BaseTrackerRecHit : public TrackingRecHit { 
public:
  BaseTrackerRecHit() {}

  virtual ~BaseTrackerRecHit() {}

  BaseTrackerRecHit( const LocalPoint& p, const LocalError&e,
		     DetId id, trackerHitRTTI::RTTI rt) :  TrackingRecHit(id,(unsigned int)(rt)), pos_(p), err_(e){}

  trackerHitRTTI::RTTI rtti() const { return trackerHitRTTI::rtti(*this);}
  bool isSingle() const { return trackerHitRTTI::isSingle(*this);}
  bool isProjected() const { return trackerHitRTTI::isProjected(*this);}
  bool isMatched() const { return trackerHitRTTI::isMatched(*this);}
  bool isMulti() const { return trackerHitRTTI::isMulti(*this);}


  // verify that hits can share clusters...
  bool sameDetModule(TrackingRecHit const & hit) const;

  virtual LocalPoint localPosition() const ;

  virtual LocalError localPositionError() const ;

  bool hasPositionAndError() const ; 
 
  const LocalPoint & localPositionFast()      const { return pos_; }
  const LocalError & localPositionErrorFast() const { return err_; }

  // to be specialized for 1D and 2D
  virtual void getKfComponents( KfComponentsHolder & holder ) const=0;
  virtual int dimension() const=0; 

  void getKfComponents1D( KfComponentsHolder & holder ) const;
  void getKfComponents2D( KfComponentsHolder & holder ) const;


public:

  // obsolete (for what tracker is concerned...) interface
  virtual AlgebraicVector parameters() const;
  virtual AlgebraicSymMatrix parametersError() const;
  virtual AlgebraicMatrix projectionMatrix() const;

 private:
  
  LocalPoint pos_;
  LocalError err_;
};

// Comparison operators
inline bool operator<( const BaseTrackerRecHit& one, const BaseTrackerRecHit& other) {
  return ( one.geographicalId() < other.geographicalId() );
}
#endif  // BaseTrackerRecHit_H
