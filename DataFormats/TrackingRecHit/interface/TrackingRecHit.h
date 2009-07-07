#ifndef TrackingRecHit_h
#define TrackingRecHit_h

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"

class TrackingRecHit {
public:

  typedef unsigned int id_type;
  
  /** Type of hits:
   *   valid    = valid hit
   *   missing  = detector is good, but no rec hit found
   *   inactive = detector is off, so there was no hope
   *   bad      = there were many bad strips within the ellipse (in Tracker)
   *            = hit is compatible with the trajectory, but chi2 is too large (in Muon System)
   */
  enum Type { valid = 0, missing = 1, inactive = 2, bad = 3 };
  /// definition of equality via shared input
  enum SharedInputType {all, some};
  
  explicit TrackingRecHit(DetId id, Type type=valid ) : m_id(id), m_status(type) {}
  explicit TrackingRecHit(id_type id=0, Type type=valid ) : m_id(id), m_status(type) {}
  
  virtual ~TrackingRecHit() {}
  
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
  
  DetId geographicalId() const {return m_id;}
  
  virtual LocalPoint localPosition() const = 0;
  
  virtual LocalError localPositionError() const = 0;
  
  virtual float weight() const {return 1.;}
  
  Type type() const { return Type(m_status); }
  Type getType() const { return Type(m_status); }
  bool isValid() const {return getType()==valid;}
  
  /** Returns true if the two TrackingRecHits are using the same input information 
   * (like Digis, Clusters, etc), false otherwise. The second argument specifies 
   * how much sharing is needed in order to return true: the value "all" 
   * means that all inputs of the two hits must be identical; the value "some" means
   * that at least one of the inputs is in common. */
  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;

protected:
  // used by muon...
  void setId(id_type iid) { m_id=iid;}
  inline void setType(Type ttype) { m_status=ttype;}

private:
  
  id_type m_id;
  int m_status;
    
};

#endif
