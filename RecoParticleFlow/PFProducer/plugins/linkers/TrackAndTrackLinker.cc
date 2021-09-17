#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class TrackAndTrackLinker : public BlockElementLinkerBase {
public:
  TrackAndTrackLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        useKDTree_(conf.getParameter<bool>("useKDTree")),
        debug_(conf.getUntrackedParameter<bool>("debug", false)) {}

  bool linkPrefilter(const reco::PFBlockElement*, const reco::PFBlockElement*) const override;

  double testLink(const reco::PFBlockElement*, const reco::PFBlockElement*) const override;

private:
  bool useKDTree_, debug_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, TrackAndTrackLinker, "TrackAndTrackLinker");

bool TrackAndTrackLinker::linkPrefilter(const reco::PFBlockElement* e1, const reco::PFBlockElement* e2) const {
  return (e1->isLinkedToDisplacedVertex() || e2->isLinkedToDisplacedVertex());
}

double TrackAndTrackLinker::testLink(const reco::PFBlockElement* elem1, const reco::PFBlockElement* elem2) const {
  constexpr reco::PFBlockElement::TrackType T_TO_DISP = reco::PFBlockElement::T_TO_DISP;
  constexpr reco::PFBlockElement::TrackType T_FROM_DISP = reco::PFBlockElement::T_FROM_DISP;
  double dist = -1.0;

  const reco::PFDisplacedTrackerVertexRef& ni1_TO_DISP = elem1->displacedVertexRef(T_TO_DISP);
  const reco::PFDisplacedTrackerVertexRef& ni2_TO_DISP = elem2->displacedVertexRef(T_TO_DISP);
  const reco::PFDisplacedTrackerVertexRef& ni1_FROM_DISP = elem1->displacedVertexRef(T_FROM_DISP);
  const reco::PFDisplacedTrackerVertexRef& ni2_FROM_DISP = elem2->displacedVertexRef(T_FROM_DISP);

  if (ni1_TO_DISP.isNonnull() && ni2_FROM_DISP.isNonnull())
    if (ni1_TO_DISP == ni2_FROM_DISP) {
      dist = 1.0;
    }

  if (ni1_FROM_DISP.isNonnull() && ni2_TO_DISP.isNonnull())
    if (ni1_FROM_DISP == ni2_TO_DISP) {
      dist = 1.0;
    }

  if (ni1_FROM_DISP.isNonnull() && ni2_FROM_DISP.isNonnull())
    if (ni1_FROM_DISP == ni2_FROM_DISP) {
      dist = 1.0;
    }

  if (elem1->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) &&
      elem2->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)) {
    for (const auto& conv1 : elem1->convRefs()) {
      for (const auto& conv2 : elem2->convRefs()) {
        if (conv1.isNonnull() && conv2.isNonnull() && conv1 == conv2) {
          dist = 1.0;
          break;
        }
      }
    }
  }

  if (elem1->trackType(reco::PFBlockElement::T_FROM_V0) && elem2->trackType(reco::PFBlockElement::T_FROM_V0)) {
    if (elem1->V0Ref().isNonnull() && elem2->V0Ref().isNonnull()) {
      if (elem1->V0Ref() == elem2->V0Ref()) {
        dist = 1.0;
      }
    }
  }

  return dist;
}
