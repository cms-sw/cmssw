#ifndef ParticleFlowCandidate_PFCandidateFwd_h
#define ParticleFlowCandidate_PFCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"

namespace reco {
  class PFCandidate;

  /// collection of PFCandidates
  typedef std::vector<reco::PFCandidate> PFCandidateCollection;

  /// iterator 
  typedef PFCandidateCollection::const_iterator PFCandidateConstIterator;

  /// iterator 
  typedef PFCandidateCollection::iterator PFCandidateIterator;

  /// persistent reference to a PFCandidate
  typedef edm::Ref<PFCandidateCollection> PFCandidateRef;

  /// persistent Ptr to a PFCandidate
  typedef edm::Ptr<PFCandidate> PFCandidatePtr;

  /// persistent reference to a PFCandidates collection
  typedef edm::RefProd<PFCandidateCollection> PFCandidateRefProd;

  /// vector of reference to GenParticleCandidate in the same collection
  typedef edm::RefVector<PFCandidateCollection> PFCandidateRefVector;

   /// persistent "forward" reference to a PFCandidate
   typedef edm::FwdRef<PFCandidateCollection> PFCandidateFwdRef;
 
   /// persistent FwdPtr to a PFCandidate
   typedef edm::FwdPtr<PFCandidate> PFCandidateFwdPtr;
 
   /// vector of "forward" reference
   typedef std::vector<PFCandidateFwdRef> PFCandidateFwdRefVector;
 
   /// vector of "forward" reference
   typedef std::vector<PFCandidateFwdPtr> PFCandidateFwdPtrVector;

  
}

#endif
