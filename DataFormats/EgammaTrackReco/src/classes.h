#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateSeedAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCandidateSuperClusterAssociation.h"
#include "DataFormats/EgammaTrackReco/interface/TrackSuperClusterAssociation.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"


#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {

    reco::TrackCandidateSuperClusterAssociationCollection v5;
    edm::Wrapper<reco::TrackCandidateSuperClusterAssociationCollection> c5;
    reco::TrackCandidateSuperClusterAssociation vv5;
    reco::TrackCandidateSuperClusterAssociationRef r5;
    reco::TrackCandidateSuperClusterAssociationRefProd rp5;
    reco::TrackCandidateSuperClusterAssociationRefVector rv5;

    reco::TrackSuperClusterAssociationCollection v6;
    edm::Wrapper<reco::TrackSuperClusterAssociationCollection> c6;
    reco::TrackSuperClusterAssociation vv6;
    reco::TrackSuperClusterAssociationRef r6;
    reco::TrackSuperClusterAssociationRefProd rp6;
    reco::TrackSuperClusterAssociationRefVector rv6;




  }
}

