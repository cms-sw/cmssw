#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include <DataFormats/Common/interface/OwnVector.h>
#include <DataFormats/Common/interface/ClonePolicy.h>

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
/* #include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h" */
/* #include "TrackingTools/TransientTrack/interface/GsfTransientTrack.h" */
#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

namespace { 
  namespace {

    std::vector<reco::PFCluster>                         dummy1;
    edm::Wrapper< std::vector<reco::PFCluster> >         dummy2;

    std::vector<reco::PFRecHit>                          dummy3;
    edm::Ref< std::vector<reco::PFRecHit> >              dummy4;
    edm::Wrapper< std::vector<reco::PFRecHit> >          dummy5;

    std::vector<reco::PFRecTrack>                        dummy6;
    edm::Wrapper< std::vector<reco::PFRecTrack> >        dummy7;

/*     edm::RefToBase<reco::Track>                          dummy7b; */
/*     edm::reftobase::BaseHolder<reco::Track>              dummy7c; */
/*     edm::reftobase::Holder<reco::GsfTrack>               dummy7d;  */
/*     edm::Holder<reco::TrackTransientTrack>               dummy7e;  */
/*     edm::Holder<reco::GsfTransientTrack>                 dummy7f; */

    std::vector<reco::PFTrajectoryPoint>                 dummy8;
    edm::Wrapper< std::vector<reco::PFTrajectoryPoint> > dummy9;

    std::vector<reco::PFSimParticle>                     dummy10;
    edm::Wrapper< std::vector<reco::PFSimParticle> >     dummy11;

    edm::OwnVector< reco::PFBlockElement, 
      edm::ClonePolicy<reco::PFBlockElement> >           dummy12;
    edm::Wrapper< edm::OwnVector< reco::PFBlockElement,
      edm::ClonePolicy< reco::PFBlockElement> >  >       dummy13;
    std::vector<reco::PFBlockElement*>                   dummy14;

    std::vector<reco::PFBlock>                           dummy15;
    edm::Wrapper< std::vector<reco::PFBlock> >           dummy16;
    edm::Ref< std::vector<reco::PFBlock>, reco::PFBlock, edm::refhelper::FindUsingAdvance< std::vector<reco::PFBlock>, reco::PFBlock> >  dummy18;
    edm::Ref<std::vector<reco::PFRecTrack>,reco::PFRecTrack,edm::refhelper::FindUsingAdvance<std::vector<reco::PFRecTrack>,reco::PFRecTrack> > dummy19;
    edm::Ref<std::vector<reco::PFCluster>,reco::PFCluster,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCluster>,reco::PFCluster> > dummy20;
  }
}
