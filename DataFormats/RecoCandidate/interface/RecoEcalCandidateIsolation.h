#ifndef EgammaCandidates_RecoEcalCandidateIsolationAssociation_h
#define EgammaCandidates_RecoEcalCandidateIsolationAssociation_h
// \class RecoEcalCandidateIsolationAssociation
// 
// \short association of Isolation to a RecoEcalCandidate
// 
// $Id: $

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include <vector>

namespace reco {
  typedef edm::AssociationMap<edm::OneToValue<std::vector<reco::RecoEcalCandidate>, float > > RecoEcalCandidateIsolationMap;
}
#endif
