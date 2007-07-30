#ifndef SiPixelRecHit_H
#define SiPixelRecHit_H

//---------------------------------------------------------------------------
//!  \class SiPixelRecHit
//!  \brief Pixel Reconstructed Hit
//!
//!  A pixel hit is a 2D position and error in a given
//!  pixel sensor. It contains a persistent reference edm::Ref to the
//!  pixel cluster. 
//!
//!  \author porting from ORCA: Petar Maksimovic (JHU), 
//!          DetSetVector and persistent references: V.Chiochia (Uni Zurich)
//---------------------------------------------------------------------------

#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Ref.h"

class SiPixelRecHit : public  BaseSiTrackerRecHit2DLocalPos {
public:

  typedef edm::Ref<edm::DetSetVector<SiPixelCluster>, SiPixelCluster > ClusterRef;

  SiPixelRecHit(): BaseSiTrackerRecHit2DLocalPos (),cluster_() {}

  ~SiPixelRecHit() {}

  SiPixelRecHit( const LocalPoint&, const LocalError&,
		 const DetId&, 
		 edm::Ref< edm::DetSetVector<SiPixelCluster>, SiPixelCluster> const&  cluster);  

  virtual SiPixelRecHit * clone() const {return new SiPixelRecHit( * this); }

  edm::Ref<edm::DetSetVector<SiPixelCluster>, 
    SiPixelCluster>  const& cluster() const { return cluster_;}

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;

private:

  edm::Ref<edm::DetSetVector<SiPixelCluster>, SiPixelCluster > cluster_;

};

// Comparison operators
inline bool operator<( const SiPixelRecHit& one, const SiPixelRecHit& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
