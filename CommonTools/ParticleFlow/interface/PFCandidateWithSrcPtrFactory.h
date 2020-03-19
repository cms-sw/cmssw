#ifndef CommonTools_ParticleFlow_PFCandidateWithSrcPtrFactor_h
#define CommonTools_ParticleFlow_PFCandidateWithSrcPtrFactor_h

/**
  \class    reco::PFCandidateWithSrcPtrFactory PFCandidateWithSrcPtrFactory.h  "CommonTools/ParticleFlow/interface/PFCandidateWithSrcPtrFactory.h"
  \brief    Creates a PFCandidate from an input FwdPtr, and sets the "source" Ptr to the FwdPtr.backPtr 


  \author   Salvatore Rappoccio
*/

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

namespace reco {
  class PFCandidateWithSrcPtrFactory {
  public:
    reco::PFCandidate operator()(edm::FwdPtr<reco::PFCandidate> const& input) const {
      reco::PFCandidate output(*input);

      if (input.backPtr().isAvailable())
        output.setSourceCandidatePtr(input.backPtr());
      else  //we are in a job where the original collection is gone
        output.setSourceCandidatePtr(input.ptr());
      return output;
    }
  };
}  // namespace reco

#endif
