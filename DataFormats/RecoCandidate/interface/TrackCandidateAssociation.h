#ifndef DataFormats_RecoCandidate_TrackCandidateAssociation_h
#define DataFormats_RecoCandidate_TrackCandidateAssociation_h
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace reco {
  typedef edm::AssociationMap<edm::OneToOne<reco::TrackCollection, reco::CandidateCollection> >
      TrackCandidateAssociation;
}

#endif
