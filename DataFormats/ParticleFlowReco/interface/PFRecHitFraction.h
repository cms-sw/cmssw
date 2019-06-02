#ifndef DataFormats_ParticleFlowReco_PFRecHitFraction_h
#define DataFormats_ParticleFlowReco_PFRecHitFraction_h

#include <iostream>
#include <vector>

#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

namespace reco {

  /**\class PFRecHitFraction
     \brief Fraction of a PFRecHit 
     (rechits can be shared between several PFCluster's)
          
     \author Colin Bernet
     \date   July 2006
  */
  class PFRecHitFraction {
  public:
    /// default constructor
    PFRecHitFraction() : fraction_(-1) {}

    /// constructor
    PFRecHitFraction(const PFRecHitRef& recHitRef, double fraction) : recHitRef_(recHitRef), fraction_(fraction) {}

    /// \return index to rechit
    const PFRecHitRef& recHitRef() const { return recHitRef_; }

    /// \return energy fraction
    double fraction() const { return fraction_; }

  private:
    /// corresponding rechit
    PFRecHitRef recHitRef_;

    /// fraction of the rechit energy owned by the cluster
    double fraction_;
  };

  std::ostream& operator<<(std::ostream& out, const PFRecHitFraction& hit);

}  // namespace reco

#endif
