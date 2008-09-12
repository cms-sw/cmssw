#ifndef ParticleFlowCandidate_PileUpPFCandidate_h
#define ParticleFlowCandidate_PileUpPFCandidate_h

#include <iostream>

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

namespace reco {
  /**\class PileUpPFCandidate
     \brief Particle reconstructed by the particle flow algorithm.
          
     \author Colin Bernet
     \date   February 2007
  */
  class PileUpPFCandidate : public PFCandidate {

  public:
    
    /// default constructor
    PileUpPFCandidate();
    
    PileUpPFCandidate( const PFCandidateRef& candidateRef);

    /// destructor
    virtual ~PileUpPFCandidate() {}

    /// return a clone
    virtual PileUpPFCandidate * clone() const;
    
/*     const PFCandidateRef& parent() const { return parent_;} */

    friend std::ostream& operator<<( std::ostream& out, 
				     const PileUpPFCandidate& c );
  
  private:
    
/*     PFCandidateRef parent_; */
  };


}

#endif
