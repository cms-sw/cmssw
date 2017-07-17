#ifndef RecoCandidates_RecoChargedCandidateIsolationAssociation_h
#define RecoCandidates_RecoChargedCandidateIsolationAssociation_h
// \class RecoChargedCandidateIsolationAssociation
// 
// \short association of Isolation to a RecoChargedCandidate
// 

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include <vector>

namespace reco {
  typedef edm::AssociationMap<edm::OneToValue<std::vector<reco::RecoChargedCandidate>, float > > RecoChargedCandidateIsolationMap;
}
#endif
