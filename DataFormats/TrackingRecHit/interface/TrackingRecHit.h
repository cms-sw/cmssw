#ifndef TrackingRecHit_h
#define TrackingRecHit_h

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#define NO_DICT
#endif

class GeomDet;
class GeomDetUnit;
class TkCloner;
class TrajectoryStateOnSurface;

class TrackingRecHit {
public:

  friend class MuonTransientTrackingRecHit;

  typedef unsigned int id_type;
  
  /** Type of hits:
   *   valid    = valid hit
   *   missing  = detector is good, but no rec hit found
   *   inactive = detector is off, so there was no hope
   *   bad      = there were many bad strips within the ellipse (in Tracker)
   *            = hit is compatible with the trajectory, but chi2 is too large (in Muon System)
   */
  enum Type { valid = 0, missing = 1, inactive = 2, bad = 3 };
  static const int typeMask = 0xf;  // mask for the above
  static const int rttiShift = 24; // shift amount to get the rtti
 
  /// definition of equality via shared input
  enum SharedInputType {all, some};
  
  explicit TrackingRecHit(DetId id, Type type=valid ) : m_id(id), m_status(type),  m_det(nullptr) {}
  explicit TrackingRecHit(id_type id=0, Type type=valid ) : m_id(id), m_status(type), m_det(nullptr) {}

  TrackingRecHit(DetId id, unsigned int rt, Type type=valid  ) : m_id(id), m_status((rt<< rttiShift)|int(type)), m_det(nullptr) {}

  TrackingRecHit(DetId id, GeomDet const * idet, Type type=valid ) : m_id(id), m_status(type),  m_det(idet) {}
  TrackingRecHit(DetId id, GeomDet const * idet, unsigned int rt, Type type=valid  ) : m_id(id), m_status((rt<< rttiShift)|int(type)), m_det(idet) {}

  TrackingRecHit(const GeomDet * idet, DetId id, Type type=valid  ) : m_id(id), m_status(type), m_det(idet){}
  TrackingRecHit(const GeomDet * idet, DetId id, unsigned int rt, Type type=valid  ) : m_id(id), m_status((rt<< rttiShift)|int(type)),  m_det(idet){}
  TrackingRecHit(const GeomDet * idet,  TrackingRecHit const & rh) : m_id(rh.m_id), m_status(rh.m_status), m_det(idet){} 


  virtual ~TrackingRecHit() {}

  // fake TTRH interface
  virtual TrackingRecHit const * hit() const { return this;}  

  
  virtual TrackingRecHit * clone() const = 0;
  
  virtual AlgebraicVector parameters() const = 0;
  
  virtual AlgebraicSymMatrix parametersError() const = 0;
  
  virtual AlgebraicMatrix projectionMatrix() const = 0;

  virtual void getKfComponents( KfComponentsHolder & holder ) const ; 
 
  virtual int dimension() const = 0;
  
  /// Access to component RecHits (if any)
  virtual std::vector<const TrackingRecHit*> recHits() const = 0;
  virtual void recHitsV(std::vector<const TrackingRecHit*> & ) const;
  
  /// Non-const access to component RecHits (if any)
  virtual std::vector<TrackingRecHit*> recHits() = 0;
  virtual void recHitsV(std::vector<TrackingRecHit*> & );
  

  id_type rawId() const { return m_id;}
  DetId geographicalId() const {return m_id;}

  const GeomDet * det() const { return m_det;}

  /// CAUTION: the GeomDetUnit* is zero for composite hits 
  /// (matched hits in the tracker, segments in the muon).
  /// Always check this pointer before using it!
  virtual const GeomDetUnit * detUnit() const;
  

  virtual LocalPoint localPosition() const = 0;
  
  virtual LocalError localPositionError() const = 0;

  /// to be redefined by daughter class
  virtual bool hasPositionAndError() const {return true;}; 
  
  virtual float weight() const {return 1.;}
  
  Type type() const { return Type(typeMask&m_status); }
  Type getType() const { return Type(typeMask&m_status); }
  bool isValid() const {return getType()==valid;}
  
  unsigned int getRTTI() const { return m_status >> rttiShift;}

  /** Returns true if the two TrackingRecHits are using the same input information 
   * (like Digis, Clusters, etc), false otherwise. The second argument specifies 
   * how much sharing is needed in order to return true: the value "all" 
   * means that all inputs of the two hits must be identical; the value "some" means
   * that at least one of the inputs is in common. */
  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;


  //  global coordinates
  
  virtual GlobalPoint globalPosition() const;
  virtual GlobalError globalPositionError() const;
  
  virtual float errorGlobalR() const;
  virtual float errorGlobalZ() const;
  virtual float errorGlobalRPhi() const;


  /// Returns true if the clone( const TrajectoryStateOnSurface&) method returns an
  /// improved hit, false if it returns an identical copy.
  /// In order to avoid redundent copies one should call canImproveWithTrack() before 
  /// calling clone( const TrajectoryStateOnSurface&).
  ///this will be done inside the TkCloner itself
  virtual bool canImproveWithTrack() const {return false;}
private:
  friend class  TkCloner;
  // double dispatch
  virtual TrackingRecHit * clone(TkCloner const&, TrajectoryStateOnSurface const&) const {
    return clone(); // default
  }

protected:
  // used by muon...
  void setId(id_type iid) { m_id=iid;}
  void setType(Type ttype) { m_status=ttype;}
  
  void setRTTI (unsigned int rt) { m_status &= (rt<< rttiShift);} // can be done only once...

private:
  
  id_type m_id;

  unsigned int m_status; // bit assigned (type 0-8) (rtti 24-31) 

  const GeomDet * m_det;

};

#endif
