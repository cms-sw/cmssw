#ifndef EgammaCandidates_RecoEcalCandidateIsolationAssociation_h
#define EgammaCandidates_RecoEcalCandidateIsolationAssociation_h
// \class RecoEcalCandidateIsolationAssociation
// 
// \short association of Isolation to a RecoEcalCandidate
// 
// $Id: RecoEcalCandidateIsolation.h,v 1.1 2006/10/20 13:08:08 rahatlou Exp $

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h" 
#include <vector>

namespace reco {
  typedef edm::AssociationMap<edm::OneToValue<std::vector<reco::RecoEcalCandidate>, float > > RecoEcalCandidateIsolationMap;
}
#endif
