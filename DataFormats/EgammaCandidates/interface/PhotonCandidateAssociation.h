#ifndef DataFormats_RecoCandidate_PhotonCandidateAssociation_h
#define DataFormats_RecoCandidate_PhotonCandidateAssociation_h
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace reco {
  typedef edm::AssociationMap<edm::OneToOne<reco::PhotonCollection, reco::CandidateCollection> > PhotonCandidateAssociation;
}

#endif
