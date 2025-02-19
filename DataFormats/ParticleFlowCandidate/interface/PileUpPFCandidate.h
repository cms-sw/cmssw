#ifndef ParticleFlowCandidate_PileUpPFCandidate_h
#define ParticleFlowCandidate_PileUpPFCandidate_h

#include <iostream>

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

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
    
    PileUpPFCandidate( const PFCandidatePtr& candidatePtr,
		       const VertexRef& vertexRef);

    /// destructor
    virtual ~PileUpPFCandidate();

    /// return a clone
    virtual PileUpPFCandidate * clone() const;
    
    /// return reference to the associated vertex
    const VertexRef&  vertexRef() const {return vertexRef_;}
    

    friend std::ostream& operator<<( std::ostream& out, 
				     const PileUpPFCandidate& c );
  
  private:
    
    VertexRef     vertexRef_;
  };


}

#endif
