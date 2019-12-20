#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class GSFAndHGCalLinker : public BlockElementLinkerBase {
public:
  GSFAndHGCalLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        useKDTree_(conf.getParameter<bool>("useKDTree")),
        debug_(conf.getUntrackedParameter<bool>("debug", false)) {}

  double testLink(const reco::PFBlockElement*, const reco::PFBlockElement*) const override;

private:
  bool useKDTree_, debug_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, GSFAndHGCalLinker, "GSFAndHGCalLinker");

double GSFAndHGCalLinker::testLink(const reco::PFBlockElement* elem1, const reco::PFBlockElement* elem2) const {
  constexpr reco::PFTrajectoryPoint::LayerType ECALShowerMax = reco::PFTrajectoryPoint::ECALShowerMax;
  const reco::PFBlockElementCluster* hgcalelem(nullptr);
  const reco::PFBlockElementGsfTrack* gsfelem(nullptr);
  double dist(-1.0);
  if (elem1->type() > elem2->type()) {
    hgcalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
    gsfelem = static_cast<const reco::PFBlockElementGsfTrack*>(elem2);
  } else {
    hgcalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
    gsfelem = static_cast<const reco::PFBlockElementGsfTrack*>(elem1);
  }
  const reco::PFRecTrack& track = gsfelem->GsftrackPF();
  const reco::PFClusterRef& clusterref = hgcalelem->clusterRef();
  const reco::PFTrajectoryPoint& tkAtECAL = track.extrapolatedPoint(ECALShowerMax);
  if (tkAtECAL.isValid()) {
    dist = LinkByRecHit::computeDist(tkAtECAL.positionREP().eta(),
                                     tkAtECAL.positionREP().phi(),
                                     clusterref->positionREP().Eta(),
                                     clusterref->positionREP().Phi());
    if (dist > 0.3)
      dist = -1.0;
  }
  if (debug_) {
    if (dist > 0.) {
      std::cout << " Here a link has been established"
                << " between a GSF track an HGCal with dist  " << dist << std::endl;
    } else {
      if (debug_)
        std::cout << " No link found " << std::endl;
    }
  }

  return dist;
}
