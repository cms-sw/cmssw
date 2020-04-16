#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class HGCalAndBREMLinker : public BlockElementLinkerBase {
public:
  HGCalAndBREMLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        useKDTree_(conf.getParameter<bool>("useKDTree")),
        debug_(conf.getUntrackedParameter<bool>("debug", false)) {}

  double testLink(const reco::PFBlockElement*, const reco::PFBlockElement*) const override;

private:
  bool useKDTree_, debug_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, HGCalAndBREMLinker, "HGCalAndBREMLinker");

double HGCalAndBREMLinker::testLink(const reco::PFBlockElement* elem1, const reco::PFBlockElement* elem2) const {
  constexpr reco::PFTrajectoryPoint::LayerType ECALShowerMax = reco::PFTrajectoryPoint::ECALShowerMax;
  const reco::PFBlockElementCluster* ecalelem(nullptr);
  const reco::PFBlockElementBrem* bremelem(nullptr);
  double dist(-1.0);
  if (elem1->type() > elem2->type()) {
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
    bremelem = static_cast<const reco::PFBlockElementBrem*>(elem2);
  } else {
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
    bremelem = static_cast<const reco::PFBlockElementBrem*>(elem1);
  }
  const reco::PFClusterRef& clusterref = ecalelem->clusterRef();
  const reco::PFRecTrack& track = bremelem->trackPF();
  const reco::PFTrajectoryPoint& tkAtECAL = track.extrapolatedPoint(ECALShowerMax);
  if (tkAtECAL.isValid()) {
    dist = LinkByRecHit::computeDist(tkAtECAL.positionREP().eta(),
                                     tkAtECAL.positionREP().phi(),
                                     clusterref->positionREP().Eta(),
                                     clusterref->positionREP().Phi());
    if (dist > 0.3)
      dist = -1.0;
  }
  return dist;
}
