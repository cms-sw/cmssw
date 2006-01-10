#ifndef SiStripRecHit2DLocalPos_H
#define SiStripRecHit2DLocalPos_H

#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

class SiStripRecHit2DLocalPos : public RecHit2DLocalPos {
public:

  SiStripRecHit2DLocalPos(): id_(0) {}

  ~SiStripRecHit2DLocalPos() {}

  SiStripRecHit2DLocalPos( const LocalPoint&, const LocalError&,
			const GeomDet*, const DetId&, 
			   const std::vector<const SiStripCluster*>& cluster);  

  virtual SiStripRecHit2DLocalPos * clone() const {return new SiStripRecHit2DLocalPos( * this); }

  virtual LocalPoint localPosition() const {return pos_;}

  virtual LocalError localPositionError() const{ return err_;}

  virtual const GeomDet& det() const {return *det_;}

  virtual DetId geographicalId() const {return id_;}

  const std::vector<const SiStripCluster*>& cluster() const { return cluster_;}
  
private:

  LocalPoint pos_;
  LocalError err_;
  const GeomDet* det_;  //! transient
  DetId id_;
  std::vector<const SiStripCluster*>      cluster_;

};

// Comparison operators
inline bool operator<( const SiStripRecHit2DLocalPos& one, const SiStripRecHit2DLocalPos& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
