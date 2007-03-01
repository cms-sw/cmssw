#ifndef DataFormats_ParticleFlowReco_PFRecHitFraction_h
#define DataFormats_ParticleFlowReco_PFRecHitFraction_h


#include <iostream>
#include <vector>

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
    PFRecHitFraction() : recHitIndex_(0), fraction_(-1) {}
    
    /// constructor
    PFRecHitFraction(unsigned rechitIndex, double fraction ) 
      : recHitIndex_(rechitIndex), fraction_(fraction) {}
    
    /// copy
    PFRecHitFraction(const PFRecHitFraction& other) 
      : recHitIndex_(other.recHitIndex_), fraction_(other.fraction_) {}
    
    /// \return index to rechit
    unsigned recHitIndex() const {return recHitIndex_;} 
						
    /// \return energy fraction
    double fraction() const {return fraction_;}
    
    friend    std::ostream& operator<<(std::ostream& out,
				       const PFRecHitFraction& hit);
    
  private:
    
    /// corresponding rechit 
    unsigned  recHitIndex_;
    
    /// fraction of the rechit energy owned by the cluster
    double    fraction_;
        
  };
}



#endif
