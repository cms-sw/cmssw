#ifndef RecoCandidate_RecoCaloJetCandidateFwd_h
#define RecoCandidate_RecoCaloJetCandidateFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <vector>

namespace reco {
  class RecoCaloJetCandidate;
  typedef std::vector<RecoCaloJetCandidate> RecoCaloJetCandidateCollection;
  typedef edm::Ref<RecoCaloJetCandidateCollection> RecoCaloJetCandidateRef;
  typedef edm::RefVector<RecoCaloJetCandidateCollection> RecoCaloJetCandidateRefVector;
  typedef edm::RefProd<RecoCaloJetCandidateCollection> RecoCaloJetCandidateRefProd;
}

#endif
