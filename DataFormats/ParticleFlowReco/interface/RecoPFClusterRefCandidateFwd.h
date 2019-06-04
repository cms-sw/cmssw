#ifndef RecoCandidate_RecoPFClusterRefCandidateFwd_h
#define RecoCandidate_RecoPFClusterRefCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/ParticleFlowReco/interface/RecoPFClusterRefCandidate.h"

namespace reco {

  /// collectin of LeafRefCandidateT<reco::TrackRef>  objects
  typedef std::vector<RecoPFClusterRefCandidate> RecoPFClusterRefCandidateCollection;

  /// reference to an object in a collection of RecoPFClusterRefCandidate objects
  typedef edm::Ref<RecoPFClusterRefCandidateCollection> RecoPFClusterRefCandidateRef;

  /// reference to a collection of RecoPFClusterRefCandidate objects
  typedef edm::RefProd<RecoPFClusterRefCandidateCollection> RecoPFClusterRefCandidateRefProd;

  /// vector of objects in the same collection of RecoPFClusterRefCandidate objects
  typedef edm::RefVector<RecoPFClusterRefCandidateCollection> RecoPFClusterRefCandidateRefVector;

  /// iterator over a vector of reference to RecoPFClusterRefCandidate objects
  typedef RecoPFClusterRefCandidateRefVector::iterator recoPFClusterRefCandidate_iterator;

  typedef edm::RefToBase<reco::Candidate> RecoPFClusterRefCandidateRefToBase;

  /*   /// this needs to go here, it's a class template in the DF/Candidate package */
  /*   /// that requires the knowledge of the DF/TrackReco dictionaries */
  /*   typedef std::vector<RecoPFClusterRefCandidateBase> RecoPFClusterRefCandidateBaseCollection; */
  /*   typedef edm::Ref<RecoPFClusterRefCandidateBaseCollection> RecoPFClusterRefCandidateBaseRef; */
  /*   typedef edm::RefVector<RecoPFClusterRefCandidateBaseCollection> RecoPFClusterRefCandidateBaseRefVector; */
  /*   typedef edm::RefProd<RecoPFClusterRefCandidateBaseCollection> RecoPFClusterRefCandidateBaseRefProd; */
  /*   typedef edm::RefToBase<reco::Candidate> RecoPFClusterRefCandidateBaseRefToBase; */

}  // namespace reco

#endif
