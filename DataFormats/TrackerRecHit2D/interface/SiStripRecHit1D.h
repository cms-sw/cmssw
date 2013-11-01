#ifndef SiStripRecHit1D_H
#define SiStripRecHit1D_H

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <atomic>
#endif

#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

class SiStripRecHit1D GCC11_FINAL : public TrackerSingleRecHit {
public:

  SiStripRecHit1D();
  // swap function
  void swap(SiStripRecHit1D& other);
  // copy ctor
  SiStripRecHit1D(const SiStripRecHit1D& src);
  // copy assignment operator
  SiStripRecHit1D& operator=(const SiStripRecHit1D& rhs);
  // move ctor
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  SiStripRecHit1D(SiStripRecHit1D&& other);
#endif

  typedef OmniClusterRef::ClusterStripRef         ClusterRef;
  typedef OmniClusterRef::ClusterRegionalRef ClusterRegionalRef;

  SiStripRecHit1D( const LocalPoint& p, const LocalError& e,
		   const DetId& id,
		   OmniClusterRef const&  clus);
  SiStripRecHit1D( const LocalPoint& p, const LocalError& e,
		   const DetId& id,
		   ClusterRef const&  clus);
  SiStripRecHit1D( const LocalPoint& p, const LocalError& e,
		   const DetId& id,
		   ClusterRegionalRef const& clus);

  /// method to facilitate the convesion from 2D to 1D hits
  SiStripRecHit1D(const SiStripRecHit2D*);

  ClusterRef cluster()  const { return cluster_strip() ; }
  void setClusterRef(ClusterRef const & ref)  {setClusterStripRef(ref);}


  virtual SiStripRecHit1D * clone() const {return new SiStripRecHit1D( * this); }


  virtual int dimension() const {return 1;}
  virtual void getKfComponents( KfComponentsHolder & holder ) const {getKfComponents1D(holder);}


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
