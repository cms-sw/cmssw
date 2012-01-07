#ifndef SiStripRecHit1D_H
#define SiStripRecHit1D_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"

#include "float.h"

class SiStripRecHit1D : public TrackingRecHit { //  public RecHit1D{
public:

  typedef TrackingRecHit Base;

  SiStripRecHit1D(): sigmaPitch_(-1.){}
  
  
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
  

  virtual int dimension() const {
    return 1;
  }

 
  ClusterRegionalRef cluster_regional()  const { 
    return cluster_.cluster_regional();
  }

  ClusterRef cluster()  const { 
    return cluster_.cluster_strip();
  }

  void setClusterRef(ClusterRef const & ref) {  cluster_ = OmniClusterRef(ref); }
  void setClusterRegionalRef(ClusterRegionalRef const & ref) { cluster_ = OmniClusterRef(ref); }



  virtual void getKfComponents( KfComponentsHolder & holder ) const ; 

  virtual LocalPoint localPosition() const;

  virtual LocalError localPositionError() const;

  bool hasPositionAndError() const ; 

  
  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
  
  double sigmaPitch() const { return sigmaPitch_;}
  void setSigmaPitch(double sigmap) const { sigmaPitch_=sigmap;}


  virtual std::vector<const TrackingRecHit*> recHits() const;
  virtual std::vector<TrackingRecHit*> recHits();


public:

  // obsolete (for what tracker is concerned...) interface
   virtual AlgebraicVector parameters() const;

  virtual AlgebraicSymMatrix parametersError() const;


   virtual AlgebraicMatrix projectionMatrix() const ;

private:
 

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
