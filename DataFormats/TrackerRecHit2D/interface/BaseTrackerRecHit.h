#ifndef BaseTrackerRecHit_H
#define BaseTrackerRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

class OmniClusterRef;

namespace trackerHitRTTI {
  // tracking hit can be : single (si1D, si2D, pix), projected, matched or multi
  enum RTTI { undef=0, single=1, proj=2, match=3, multi=4};
  inline RTTI rtti(TrackingRecHit const & hit)  { return RTTI(hit.getRTTI());}
  inline bool isUndef(TrackingRecHit const & hit) { return rtti(hit)==undef;}
  inline bool isSingle(TrackingRecHit const & hit)  { return rtti(hit)==single;}
  inline bool isProjected(TrackingRecHit const & hit)  { return rtti(hit)==proj;}
  inline bool isMatched(TrackingRecHit const & hit)  { return rtti(hit)==match;}
  inline bool isMulti(TrackingRecHit const & hit)  { return rtti(hit)==multi;}
}

class BaseTrackerRecHit : public TrackingRecHit { 
public:
  BaseTrackerRecHit() {}

  virtual ~BaseTrackerRecHit() {}

  // no position (as in persistent)
  BaseTrackerRecHit(DetId id, trackerHitRTTI::RTTI rt) :  TrackingRecHit(id,(unsigned int)(rt)) {}

  BaseTrackerRecHit( const LocalPoint& p, const LocalError&e,
		     DetId id, trackerHitRTTI::RTTI rt) :  TrackingRecHit(id,(unsigned int)(rt)), pos_(p), err_(e){}

  trackerHitRTTI::RTTI rtti() const { return trackerHitRTTI::rtti(*this);}
  bool isSingle() const { return trackerHitRTTI::isSingle(*this);}
  bool isProjected() const { return trackerHitRTTI::isProjected(*this);}
  bool isMatched() const { return trackerHitRTTI::isMatched(*this);}
  bool isMulti() const { return trackerHitRTTI::isMulti(*this);}

 // used by trackMerger (to be improved)
  virtual OmniClusterRef const & firstClusterRef() const=0;


  // verify that hits can share clusters...
  inline bool sameDetModule(TrackingRecHit const & hit) const;

  virtual LocalPoint localPosition() const  GCC11_FINAL;

  virtual LocalError localPositionError() const  GCC11_FINAL;

  bool hasPositionAndError() const  GCC11_FINAL; 
 
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


bool BaseTrackerRecHit::sameDetModule(TrackingRecHit const & hit) const {
  unsigned int myid = geographicalId().rawId();
  unsigned int mysubd = myid >> (DetId::kSubdetOffset);

  unsigned int id = hit.geographicalId().rawId();
  unsigned int subd = id >> (DetId::kSubdetOffset);
  
  if (mysubd!=subd) return false;
  
  //Protection against invalid hits
  if(!hit.isValid()) return false;
  
  const unsigned int limdet = 10;  // TIB=11
  
  if (mysubd>limdet) { // strip
    // mask glue and stereo
    myid|=3;
    id|=3;
  }
  return id==myid;

}


// Comparison operators
inline bool operator<( const BaseTrackerRecHit& one, const BaseTrackerRecHit& other) {
  return ( one.geographicalId() < other.geographicalId() );
}
#endif  // BaseTrackerRecHit_H
