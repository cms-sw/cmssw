#ifndef ParticleFlowCandidate_PFCandidateElectronExtraFwd_h
#define ParticleFlowCandidate_PFCandidateElectronExtraFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"

namespace reco {
  class PFCandidateElectronExtra;

  /// collection of PFCandidateElectronExtras
  typedef std::vector<reco::PFCandidateElectronExtra> PFCandidateElectronExtraCollection;

  /// iterator
  typedef PFCandidateElectronExtraCollection::const_iterator PFCandidateElectronExtraConstIterator;

  /// iterator
  typedef PFCandidateElectronExtraCollection::iterator PFCandidateElectronExtraIterator;

  /// persistent reference to a PFCandidateElectronExtra
  typedef edm::Ref<PFCandidateElectronExtraCollection> PFCandidateElectronExtraRef;

  /// persistent Ptr to a PFCandidateElectronExtra
  typedef edm::Ptr<PFCandidateElectronExtra> PFCandidateElectronExtraPtr;

  /// persistent reference to a PFCandidateElectronExtras collection
  typedef edm::RefProd<PFCandidateElectronExtraCollection> PFCandidateElectronExtraRefProd;

  /// vector of reference to GenParticleCandidate in the same collection
  typedef edm::RefVector<PFCandidateElectronExtraCollection> PFCandidateElectronExtraRefVector;

  /// persistent "forward" reference to a PFCandidateElectronExtra
  typedef edm::FwdRef<PFCandidateElectronExtraCollection> PFCandidateElectronExtraFwdRef;

  /// persistent FwdPtr to a PFCandidateElectronExtra
  typedef edm::FwdPtr<PFCandidateElectronExtra> PFCandidateElectronExtraFwdPtr;

  /// vector of "forward" reference
  typedef std::vector<PFCandidateElectronExtraFwdRef> PFCandidateElectronExtraFwdRefVector;

  /// vector of "forward" reference
  typedef std::vector<PFCandidateElectronExtraFwdPtr> PFCandidateElectronExtraFwdPtrVector;

}  // namespace reco
#endif
