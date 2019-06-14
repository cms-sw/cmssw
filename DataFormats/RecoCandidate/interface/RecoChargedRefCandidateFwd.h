#ifndef RecoCandidate_RecoChargedRefCandidateFwd_h
#define RecoCandidate_RecoChargedRefCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedRefCandidate.h"

namespace reco {

  /// collectin of LeafRefCandidateT<reco::TrackRef>  objects
  typedef std::vector<RecoChargedRefCandidate> RecoChargedRefCandidateCollection;

  /// reference to an object in a collection of RecoChargedRefCandidate objects
  typedef edm::Ref<RecoChargedRefCandidateCollection> RecoChargedRefCandidateRef;

  /// reference to a collection of RecoChargedRefCandidate objects
  typedef edm::RefProd<RecoChargedRefCandidateCollection> RecoChargedRefCandidateRefProd;

  /// vector of objects in the same collection of RecoChargedRefCandidate objects
  typedef edm::RefVector<RecoChargedRefCandidateCollection> RecoChargedRefCandidateRefVector;

  /// iterator over a vector of reference to RecoChargedRefCandidate objects
  typedef RecoChargedRefCandidateRefVector::iterator recoChargedRefCandidate_iterator;

  typedef edm::RefToBase<reco::Candidate> RecoChargedRefCandidateRefToBase;

  /*   /// this needs to go here, it's a class template in the DF/Candidate package */
  /*   /// that requires the knowledge of the DF/TrackReco dictionaries */
  /*   typedef std::vector<RecoChargedRefCandidateBase> RecoChargedRefCandidateBaseCollection; */
  /*   typedef edm::Ref<RecoChargedRefCandidateBaseCollection> RecoChargedRefCandidateBaseRef; */
  /*   typedef edm::RefVector<RecoChargedRefCandidateBaseCollection> RecoChargedRefCandidateBaseRefVector; */
  /*   typedef edm::RefProd<RecoChargedRefCandidateBaseCollection> RecoChargedRefCandidateBaseRefProd; */
  /*   typedef edm::RefToBase<reco::Candidate> RecoChargedRefCandidateBaseRefToBase; */

}  // namespace reco

#endif
