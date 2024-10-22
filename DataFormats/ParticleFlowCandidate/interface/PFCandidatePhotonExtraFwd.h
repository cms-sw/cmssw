#ifndef ParticleFlowCandidate_PFCandidatePhotonExtraFwd_h
#define ParticleFlowCandidate_PFCandidatePhotonExtraFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"

namespace reco {
  class PFCandidatePhotonExtra;

  /// collection of PFCandidatePhotonExtras
  typedef std::vector<reco::PFCandidatePhotonExtra> PFCandidatePhotonExtraCollection;

  /// persistent reference to a PFCandidatePhotonExtra
  typedef edm::Ref<PFCandidatePhotonExtraCollection> PFCandidatePhotonExtraRef;

  /// persistent reference to a PFCandidatePhotonExtras collection
  typedef edm::RefProd<PFCandidatePhotonExtraCollection> PFCandidatePhotonExtraRefProd;

  /*
    /// persistent Ptr to a PFCandidatePhotonExtra
    typedef edm::Ptr<PFCandidatePhotonExtra> PFCandidatePhotonExtraPtr;
    
    
    
    /// vector of reference to GenParticleCandidate in the same collection
    typedef edm::RefVector<PFCandidatePhotonExtraCollection> PFCandidatePhotonExtraRefVector;
    
    /// persistent "forward" reference to a PFCandidatePhotonExtra
    typedef edm::FwdRef<PFCandidatePhotonExtraCollection> PFCandidatePhotonExtraFwdRef;
    
    /// persistent FwdPtr to a PFCandidatePhotonExtra
    typedef edm::FwdPtr<PFCandidatePhotonExtra> PFCandidatePhotonExtraFwdPtr;
    
    /// vector of "forward" reference
    typedef std::vector<PFCandidatePhotonExtraFwdRef> PFCandidatePhotonExtraFwdRefVector;
    
    /// vector of "forward" reference
    typedef std::vector<PFCandidatePhotonExtraFwdPtr> PFCandidatePhotonExtraFwdPtrVector;
  */
}  // namespace reco
#endif
