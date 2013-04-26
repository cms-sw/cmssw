#ifndef ParticleFlowCandidate_PFCandidateEGammaExtraFwd_h
#define ParticleFlowCandidate_PFCandidateEGammaExtraFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"

namespace reco {
  class PFCandidateEGammaExtra;

    /// collection of PFCandidateEGammaExtras
  typedef std::vector<reco::PFCandidateEGammaExtra> PFCandidateEGammaExtraCollection;

  /// iterator 
  typedef PFCandidateEGammaExtraCollection::const_iterator PFCandidateEGammaExtraConstIterator;

  /// iterator 
  typedef PFCandidateEGammaExtraCollection::iterator PFCandidateEGammaExtraIterator;

  /// persistent reference to a PFCandidateEGammaExtra
  typedef edm::Ref<PFCandidateEGammaExtraCollection> PFCandidateEGammaExtraRef;

  /// persistent Ptr to a PFCandidateEGammaExtra
  typedef edm::Ptr<PFCandidateEGammaExtra> PFCandidateEGammaExtraPtr;

  /// persistent reference to a PFCandidateEGammaExtras collection
  typedef edm::RefProd<PFCandidateEGammaExtraCollection> PFCandidateEGammaExtraRefProd;

  /// vector of reference to GenParticleCandidate in the same collection
  typedef edm::RefVector<PFCandidateEGammaExtraCollection> PFCandidateEGammaExtraRefVector;

   /// persistent "forward" reference to a PFCandidateEGammaExtra
   typedef edm::FwdRef<PFCandidateEGammaExtraCollection> PFCandidateEGammaExtraFwdRef;
 
   /// persistent FwdPtr to a PFCandidateEGammaExtra
   typedef edm::FwdPtr<PFCandidateEGammaExtra> PFCandidateEGammaExtraFwdPtr;
 
   /// vector of "forward" reference
   typedef std::vector<PFCandidateEGammaExtraFwdRef> PFCandidateEGammaExtraFwdRefVector;
 
   /// vector of "forward" reference
   typedef std::vector<PFCandidateEGammaExtraFwdPtr> PFCandidateEGammaExtraFwdPtrVector;

}
#endif
