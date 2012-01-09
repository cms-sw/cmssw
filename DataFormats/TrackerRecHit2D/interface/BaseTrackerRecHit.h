#ifndef BaseTrackerRecHit_H
#define BaseTrackerRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"


class BaseTrackerRecHit : public TrackingRecHit { 
public:

  BaseTrackerRecHit() {}

  virtual ~BaseTrackerRecHit() {}

  BaseTrackerRecHit( const LocalPoint& p, const LocalError&e,
				 DetId id) :  TrackingRecHit(id), pos_(p), err_(e){}


  // verify that hits can share clusters...
  bool sameDetModule() const;

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
