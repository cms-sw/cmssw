#ifndef DataFormats_ParticleFlowReco_PFRecHitFraction_h
#define DataFormats_ParticleFlowReco_PFRecHitFraction_h

#include "Math/GenVector/PositionVector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include <iostream>
#include <vector>

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"


namespace reco {
  

  /**\class PFRecHitFraction
     \brief Fraction of a PFRecHit (rechits can be shared between several PFCluster's)
          
     \author Colin Bernet
     \date   July 2006
  */
  class PFRecHitFraction {
  public:
    
    /// default constructor
    PFRecHitFraction() : recHit_(0), fraction_(-1), distance_(0) {}
    
    /// constructor
    PFRecHitFraction(const reco::PFRecHit* rechit, double fraction, double dist) 
      : recHit_(rechit), fraction_(fraction), distance_(dist) {}
    
    /// constructor
    PFRecHitFraction(const reco::PFRecHit* rechit, double fraction) 
      : recHit_(rechit), fraction_(fraction), distance_(0) {}
    
    /// copy
    PFRecHitFraction(const PFRecHitFraction& other) 
      : recHit_(other.recHit_), fraction_(other.fraction_), distance_(other.distance_) {}
    
    /// \return pointer to rechit
    const reco::PFRecHit* getRecHit() const {return recHit_;} 
    
    /// sets distance to cluster
    void   setDistToCluster(double dist) { distance_ = dist;}
    
    /// \return energy fraction
    double getFraction() const {return fraction_;}
    
    /// \return recHit_->energy() * fraction_
    double energy() const 
      { return recHit_->energy() * fraction_;}
    
    /// \return distance to cluster
    double getDistToCluster() const {return distance_;}
    
    friend    std::ostream& operator<<(std::ostream& out,
				       const PFRecHitFraction& hit);
    
  private:
    
    /// corresponding rechit (not owner)
    const reco::PFRecHit* recHit_;
    
    /// fraction of the rechit energy owned by the cluster
    double    fraction_;
    
    /// distance to the cluster
    double    distance_;
    
  };
}



#endif
