#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class GSFAndBREMLinker : public BlockElementLinkerBase {
public:
  GSFAndBREMLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        useKDTree_(conf.getParameter<bool>("useKDTree")),
        debug_(conf.getUntrackedParameter<bool>("debug", false)) {}

  double testLink(size_t ielem1,
                  size_t ielem2,
                  reco::PFBlockElement::Type type1,
                  reco::PFBlockElement::Type type2,
                  const ElementListConst& elements,
                  const PFTables& tables,
                  const reco::PFMultiLinksIndex& multilinks) const override;

private:
  bool useKDTree_, debug_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, GSFAndBREMLinker, "GSFAndBREMLinker");

double GSFAndBREMLinker::testLink(size_t ielem1,
                                  size_t ielem2,
                                  reco::PFBlockElement::Type type1,
                                  reco::PFBlockElement::Type type2,
                                  const ElementListConst& elements,
                                  const PFTables& tables,
                                  const reco::PFMultiLinksIndex& multilinks) const {
  const auto* elem1 = elements[ielem1];
  const auto* elem2 = elements[ielem2];

  double dist = -1.0;
  const reco::PFBlockElementGsfTrack* gsfelem(nullptr);
  const reco::PFBlockElementBrem* bremelem(nullptr);
  if (type1 < type2) {
    gsfelem = static_cast<const reco::PFBlockElementGsfTrack*>(elem1);
    bremelem = static_cast<const reco::PFBlockElementBrem*>(elem2);
  } else {
    gsfelem = static_cast<const reco::PFBlockElementGsfTrack*>(elem2);
    bremelem = static_cast<const reco::PFBlockElementBrem*>(elem1);
  }
  const reco::GsfPFRecTrackRef& gsfref = gsfelem->GsftrackRefPF();
  const reco::GsfPFRecTrackRef& bremref = bremelem->GsftrackRefPF();
  if (gsfref.isNonnull() && bremref.isNonnull() && gsfref == bremref) {
    dist = 0.001;
  }
  return dist;
}
