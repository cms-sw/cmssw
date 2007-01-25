#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "DataFormats/ParticleFlowReco/interface/PFResolutionMap.h"

namespace { 
  namespace {

    std::vector<reco::PFCluster>         tsv1;
    edm::Wrapper< std::vector<reco::PFCluster> >         PFClusterProd;
    std::vector<reco::PFRecHit>          tsv2;
    edm::Wrapper< std::vector<reco::PFRecHit> >          PFHitProd;
    std::vector<reco::PFRecTrack>        tsv3;
    edm::Wrapper< std::vector<reco::PFRecTrack> >        PFRecTrackProd;
    std::vector<reco::PFTrajectoryPoint> tsv4;
    edm::Wrapper< std::vector<reco::PFTrajectoryPoint> > PFTrajPtProd;
    std::vector<reco::PFParticle>        tsv5;
    edm::Wrapper< std::vector<reco::PFParticle> >        PFParticleProd;
  }
}
