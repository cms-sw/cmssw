#ifndef SiStripRecHit2D_H
#define SiStripRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"


class SiStripRecHit2D GCC11_FINAL : public TrackerSingleRecHit {
public:

  SiStripRecHit2D(): sigmaPitch_(-1.){}

  ~SiStripRecHit2D() {} 

  typedef OmniClusterRef::ClusterStripRef         ClusterRef;
  typedef OmniClusterRef::ClusterRegionalRef ClusterRegionalRef;

  // no position (as in persistent)
  SiStripRecHit2D(const DetId& id,
		  OmniClusterRef const& clus) : 
    TrackerSingleRecHit(id, clus),
    sigmaPitch_(-1.) {}


  SiStripRecHit2D( const LocalPoint& pos, const LocalError& err, float isigmaPitch,
		   const DetId& id,
		   OmniClusterRef const& clus) : 
    TrackerSingleRecHit(pos,err,id, clus),
    sigmaPitch_(isigmaPitch) {}
 
  SiStripRecHit2D( const LocalPoint& pos, const LocalError& err, float isigmaPitch,
		   const DetId& id,
		   ClusterRef const& clus) : 
    TrackerSingleRecHit(pos,err,id, clus),
    sigmaPitch_(isigmaPitch) {}


  SiStripRecHit2D(const LocalPoint& pos, const LocalError& err, float isigmaPitch,
		  const DetId& id,
		  ClusterRegionalRef const& clus) : 
    TrackerSingleRecHit(pos,err,id, clus),
    sigmaPitch_(isigmaPitch) {}

				
  ClusterRef cluster()  const { return cluster_strip() ; }
  void setClusterRef(ClusterRef const & ref)  {setClusterStripRef(ref);}

  virtual SiStripRecHit2D * clone() const {return new SiStripRecHit2D( * this); }
  
  virtual int dimension() const {return 2;}
  virtual void getKfComponents( KfComponentsHolder & holder ) const { getKfComponents2D(holder); }

 
  float sigmaPitch() const { return sigmaPitch_;}

  
private:

  /// cache for the matcher....
  float sigmaPitch_;  // transient....

 
};

#endif
