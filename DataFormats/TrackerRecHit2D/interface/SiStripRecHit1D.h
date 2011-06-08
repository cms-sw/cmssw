#ifndef SiStripRecHit1D_H
#define SiStripRecHit1D_H

#include "DataFormats/TrackingRecHit/interface/RecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"

#include "float.h"

class SiStripRecHit1D : public RecHit1D{
public:

  SiStripRecHit1D(): RecHit1D(),
		     sigmaPitch_(-1.){}
  
  
  typedef OmniClusterRef::ClusterRef         ClusterRef;
  typedef OmniClusterRef::ClusterRegionalRef ClusterRegionalRef;


  SiStripRecHit1D( const LocalPoint&, const LocalError&,
		   const DetId&, 
		   ClusterRef const&  cluster); 

  SiStripRecHit1D( const LocalPoint&, const LocalError&,
		   const DetId&, 
		   ClusterRegionalRef const& cluster);
  
  /// method to facilitate the convesion from 2D to 1D hits
  SiStripRecHit1D(const SiStripRecHit2D*);

  virtual SiStripRecHit1D * clone() const {return new SiStripRecHit1D( * this); }
  

 
  ClusterRegionalRef cluster_regional()  const { 
    return cluster_.cluster_regional();
  }

  ClusterRef cluster()  const { 
    return cluster_.cluster();
  }

 
  virtual void getKfComponents( KfComponentsHolder & holder ) const ; 

  virtual LocalPoint localPosition() const;

  virtual LocalError localPositionError() const;

  bool hasPositionAndError() const ; 

  
  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
  
  double sigmaPitch() const { return sigmaPitch_;}
  void setSigmaPitch(double sigmap) const { sigmaPitch_=sigmap;}


  virtual std::vector<const TrackingRecHit*> recHits() const;
  virtual std::vector<TrackingRecHit*> recHits();


 private:
  void throwExceptionUninitialized(const char *where) const;
 

 /// cache for the matcher....
  mutable double sigmaPitch_;  // transient....
 
  LocalPoint pos_;
  LocalError err_;

 
 // new game
  OmniClusterRef cluster_;
 
};

// Comparison operators
inline bool operator<( const SiStripRecHit1D& one, const SiStripRecHit1D& other) {
  return one.geographicalId() < other.geographicalId() ;
}

#endif
