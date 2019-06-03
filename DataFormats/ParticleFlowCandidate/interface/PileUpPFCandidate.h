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

    PileUpPFCandidate(const PFCandidatePtr& candidatePtr, const VertexRef& vertexRef);

    /// destructor
    ~PileUpPFCandidate() override;

    /// return a clone
    PileUpPFCandidate* clone() const override;

    /// return reference to the associated vertex
    const VertexRef& vertexRef() const { return vertexRef_; }

  private:
    VertexRef vertexRef_;
  };

  std::ostream& operator<<(std::ostream& out, const PileUpPFCandidate& c);

}  // namespace reco

#endif
