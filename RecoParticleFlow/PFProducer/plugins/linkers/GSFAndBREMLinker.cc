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
  using GsfTrackRefPFIsNonNull = edm::soa::col::pf::track::GsfTrackRefPFIsNonNull;
  using GsfTrackRefPFKey = edm::soa::col::pf::track::GsfTrackRefPFKey;
  double dist = -1.0;

  size_t igsf_elem;
  size_t ibrem_elem;

  if (type1 < type2) {
    igsf_elem = ielem1;
    ibrem_elem = ielem2;

  } else {
    igsf_elem = ielem2;
    ibrem_elem = ielem1;
  }

  const size_t igsf = tables.element_to_gsf[igsf_elem];
  const size_t ibrem = tables.element_to_brem[ibrem_elem];

  const auto gr_nn = tables.gsf_table.get<GsfTrackRefPFIsNonNull>(igsf);
  const auto br_nn = tables.brem_table.get<GsfTrackRefPFIsNonNull>(ibrem);
  const auto gr_k = tables.gsf_table.get<GsfTrackRefPFKey>(igsf);
  const auto br_k = tables.brem_table.get<GsfTrackRefPFKey>(ibrem);

  if (gr_nn && br_nn && gr_k == br_k) {
    dist = 0.001;
  }
  return dist;
}
