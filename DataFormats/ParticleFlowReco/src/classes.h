#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"

namespace { 
  namespace {

    std::vector<reco::PFCluster>          tsv3;
    edm::Wrapper< std::vector<reco::PFCluster> >         PFClusterProd;
    std::vector<reco::PFRecHit>          tsv4;
    edm::Wrapper< std::vector<reco::PFRecHit> >          PFHitProd;
    std::vector<reco::PFRecTrack>        tsv5;
    edm::Wrapper< std::vector<reco::PFRecTrack> >        PFRecTrackProd;
    std::vector<reco::PFTrajectoryPoint> tsv6;
    edm::Wrapper< std::vector<reco::PFTrajectoryPoint> > PFTrajPtProd;
  }
}
