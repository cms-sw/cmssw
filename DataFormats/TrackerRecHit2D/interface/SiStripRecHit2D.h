#ifndef SiStripRecHit2D_H
#define SiStripRecHit2D_H

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <atomic>
#endif
#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"


class SiStripRecHit2D GCC11_FINAL : public TrackerSingleRecHit {
public:

  SiStripRecHit2D();
  ~SiStripRecHit2D() {}
  // swap function
  void swap(SiStripRecHit2D& other);
  // copy ctor
  SiStripRecHit2D(const SiStripRecHit2D& src);
  // copy assignment operator
  SiStripRecHit2D& operator=(const SiStripRecHit2D& rhs);
  // move ctor
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  SiStripRecHit2D(SiStripRecHit2D&& other);
#endif

  typedef OmniClusterRef::ClusterStripRef         ClusterRef;
  typedef OmniClusterRef::ClusterRegionalRef ClusterRegionalRef;

  // no position (as in persistent)
  SiStripRecHit2D(const DetId& id, OmniClusterRef const& clus);
  SiStripRecHit2D( const LocalPoint& pos, const LocalError& err,
		   const DetId& id,
		   OmniClusterRef const& clus);
  SiStripRecHit2D( const LocalPoint& pos, const LocalError& err,
		   const DetId& id,
		   ClusterRef const& clus);
  SiStripRecHit2D(const LocalPoint& pos, const LocalError& err,
		  const DetId& id,
		  ClusterRegionalRef const& clus);

  ClusterRef cluster()  const { return cluster_strip() ; }
  void setClusterRef(ClusterRef const & ref)  {setClusterStripRef(ref);}

  virtual SiStripRecHit2D * clone() const {return new SiStripRecHit2D( * this); }

  virtual int dimension() const {return 2;}
  virtual void getKfComponents( KfComponentsHolder & holder ) const { getKfComponents2D(holder); }


  double sigmaPitch() const;
  void setSigmaPitch(double sigmap) const;

private:

  /// cache for the matcher....
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  mutable std::atomic<double> sigmaPitch_;  // transient....
#else
  mutable double sigmaPitch_;  // transient....
#endif


};

#endif
