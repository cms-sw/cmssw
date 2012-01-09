#ifndef BaseTrackerRecHit_H
#define BaseTrackerRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"


class BaseTrackerRecHit : public TrackingRecHit { 
public:
  // tracking hit can be : single (si1D, si2D, pix), projected, matched or multi
  enum RTTI { undef=0, single=1, proj=2, match=3};
  BaseTrackerRecHit() {}

  virtual ~BaseTrackerRecHit() {}

  BaseTrackerRecHit( const LocalPoint& p, const LocalError&e,
		     DetId id, RTTI rt=undef) :  TrackingRecHit(id,unsigned int(rt)), pos_(p), err_(e){}

  RTTI rtti() const { return RTTI(getRTTI());}
  bool isSingle() const { return rtti()=single;}
  bool isProjected() const { return rtti()=proj;}
  bool isMatched() const { return rtti()=match;}


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
