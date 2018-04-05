#ifndef ParticleFlowCandidate_IsolatedPFCandidate_h
#define ParticleFlowCandidate_IsolatedPFCandidate_h

#include <iostream>

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

namespace reco {
  /**\class IsolatedPFCandidate
     \brief Particle reconstructed by the particle flow algorithm.
          
     \author Colin Bernet
     \date   February 2007
  */
  class IsolatedPFCandidate : public PFCandidate {

  public:


    /// default constructor
    IsolatedPFCandidate();
    
    IsolatedPFCandidate( const PFCandidatePtr& candidatePtr, 
			 double isolation );

    /// destructor
    ~IsolatedPFCandidate() override;

    /// return a clone
    IsolatedPFCandidate * clone() const override;
    
/*     const PFCandidateRef& parent() const { return parent_;} */

    double isolation() const { return isolation_; }
    
  private:

/*     PFCandidateRef parent_; */

    double isolation_;
  };

  std::ostream& operator<<( std::ostream& out, 
                            const IsolatedPFCandidate& c );
  


}

#endif
