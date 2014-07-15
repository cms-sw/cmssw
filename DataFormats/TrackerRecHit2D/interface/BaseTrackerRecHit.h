#ifndef BaseTrackerRecHit_H
#define BaseTrackerRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitGlobalState.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GeometrySurface/interface/Surface.h" 



// #define VI_DEBUG

class OmniClusterRef;

namespace trackerHitRTTI {
  // tracking hit can be : single (si1D, si2D, pix), projected, matched or multi
  enum RTTI { undef=0, single=1, projStereo=2, projMono=3, match=4, multi=5, gs=6};
  inline RTTI rtti(TrackingRecHit const & hit)  { return RTTI(hit.getRTTI());}
  inline bool isUndef(TrackingRecHit const & hit) { return rtti(hit)==undef;}
  inline bool isSingle(TrackingRecHit const & hit)  { return rtti(hit)==single;}
  inline bool isProjMono(TrackingRecHit const & hit)  { return rtti(hit)==projMono;}
  inline bool isProjStereo(TrackingRecHit const & hit)  { return rtti(hit)==projStereo;}
  inline bool isProjected(TrackingRecHit const & hit)  { return (rtti(hit)==projMono) | (rtti(hit)==projStereo);}
  inline bool isMatched(TrackingRecHit const & hit)  { return rtti(hit)==match;}
  inline bool isMulti(TrackingRecHit const & hit)  { return rtti(hit)==multi;}
  inline bool isGS(TrackingRecHit const & hit)  { return rtti(hit)==gs;}
  inline bool isSingleType(TrackingRecHit const & hit)  { return (rtti(hit)>0) & (rtti(hit)<4) ;}

  inline unsigned int  projId(TrackingRecHit const & hit) { return hit.rawId()+int(rtti(hit))-1;}
}

class BaseTrackerRecHit : public TrackingRecHit { 
public:
  BaseTrackerRecHit() : qualWord_(0){}

  // fake TTRH interface
  BaseTrackerRecHit const * hit() const GCC11_FINAL { return this;}  

  virtual ~BaseTrackerRecHit() {}

  // no position (as in persistent)
  BaseTrackerRecHit(DetId id, trackerHitRTTI::RTTI rt) :  TrackingRecHit(id,(unsigned int)(rt)) {}

  BaseTrackerRecHit( const LocalPoint& p, const LocalError&e,
		     GeomDet const & idet, trackerHitRTTI::RTTI rt) :  
    TrackingRecHit(idet, (unsigned int)(rt)), pos_(p), err_(e){
    LocalError lape = det()->localAlignmentError();
    if (lape.valid())
      err_ = LocalError(err_.xx()+lape.xx(),
			err_.xy()+lape.xy(),
			err_.yy()+lape.yy()
			);
  }

  trackerHitRTTI::RTTI rtti() const { return trackerHitRTTI::rtti(*this);}
  bool isSingle() const { return trackerHitRTTI::isSingle(*this);}
  bool isMatched() const { return trackerHitRTTI::isMatched(*this);}
  bool isProjected() const { return trackerHitRTTI::isProjected(*this);}
  bool isProjMono() const { return trackerHitRTTI::isProjMono(*this);}
  bool isProjSterep() const { return trackerHitRTTI::isProjStereo(*this);}
  bool isMulti() const { return trackerHitRTTI::isMulti(*this);}

  virtual bool isPixel() const { return false;}

 // used by trackMerger (to be improved)
  virtual OmniClusterRef const & firstClusterRef() const=0;


  // verify that hits can share clusters...
  inline bool sameDetModule(TrackingRecHit const & hit) const;

  bool hasPositionAndError() const  GCC11_FINAL; 

  virtual LocalPoint localPosition() const  GCC11_FINAL { check(); return pos_;}

  virtual LocalError localPositionError() const  GCC11_FINAL { check(); return err_;}

 
  const LocalPoint & localPositionFast()      const { check(); return pos_; }
  const LocalError & localPositionErrorFast() const { check(); return err_; }



  // to be specialized for 1D and 2D
  virtual void getKfComponents( KfComponentsHolder & holder ) const=0;
  virtual int dimension() const=0; 

  void getKfComponents1D( KfComponentsHolder & holder ) const;
  void getKfComponents2D( KfComponentsHolder & holder ) const;


  // global coordinates
  // Extension of the TrackingRecHit interface
  virtual const Surface * surface() const GCC11_FINAL {return &(det()->surface());}


  virtual GlobalPoint globalPosition() const GCC11_FINAL {
      return surface()->toGlobal(localPosition());
  }
  
  GlobalError globalPositionError() const GCC11_FINAL { return ErrorFrameTransformer().transform( localPositionError(), *surface() );}
  float errorGlobalR() const GCC11_FINAL { return std::sqrt(globalPositionError().rerr(globalPosition()));}
  float errorGlobalZ() const GCC11_FINAL { return std::sqrt(globalPositionError().czz()); }
  float errorGlobalRPhi() const GCC11_FINAL { return globalPosition().perp()*sqrt(globalPositionError().phierr(globalPosition())); }

  // once cache removed will obsolete the above
  TrackingRecHitGlobalState globalState() const {
    GlobalError  
      globalError = ErrorFrameTransformer::transform( localPositionError(), *surface() );
    GlobalPoint gp = globalPosition();
    float r = gp.perp();
    float errorRPhi = r*std::sqrt(float(globalError.phierr(gp))); 
    float errorR = std::sqrt(float(globalError.rerr(gp)));
    float errorZ = std::sqrt(float(globalError.czz()));
    return (TrackingRecHitGlobalState){
      gp.basicVector(), r, gp.barePhi(),
	errorR,errorZ,errorRPhi
	};
  }

  /// cluster probability, overloaded by pixel rechits.
  virtual float clusterProbability() const { return 1.f; }

public:

  // obsolete (for what tracker is concerned...) interface
  virtual AlgebraicVector parameters() const;
  virtual AlgebraicSymMatrix parametersError() const;
  virtual AlgebraicMatrix projectionMatrix() const;

private:

#ifdef VI_DEBUG
  void check() const { assert(det());}
#elif EDM_LM_DEBUG
  void check() const;
#else 
  static void check(){}
#endif

private:

  LocalPoint pos_;
  LocalError err_;
protected:
  //this comes for free (padding)
  unsigned int qualWord_;
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
