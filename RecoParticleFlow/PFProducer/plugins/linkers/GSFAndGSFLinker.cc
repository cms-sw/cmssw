#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class GSFAndGSFLinker : public BlockElementLinkerBase {
public:
  GSFAndGSFLinker(const edm::ParameterSet& conf)
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

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, GSFAndGSFLinker, "GSFAndGSFLinker");

double GSFAndGSFLinker::testLink(size_t ielem1,
                                 size_t ielem2,
                                 reco::PFBlockElement::Type type1,
                                 reco::PFBlockElement::Type type2,
                                 const ElementListConst& elements,
                                 const PFTables& tables,
                                 const reco::PFMultiLinksIndex& multilinks) const {
  using is_nn = edm::soa::col::pf::track::GsfTrackRefPFIsNonNull;
  using from_gconv = edm::soa::col::pf::track::GsfTrackRefPFIsNonNull;
  using tid = edm::soa::col::pf::track::GsfTrackRefPFTrackId;

  double dist = -1.0;

  const size_t igsf1 = tables.element_to_gsf[ielem1];
  const size_t igsf2 = tables.element_to_gsf[ielem2];

  const auto& gsft = tables.gsf_table;

  if (gsft.get<is_nn>(igsf1) && gsft.get<is_nn>(igsf2)) {
    if (gsft.get<from_gconv>(igsf1) !=  // we want **one** primary GSF
            gsft.get<from_gconv>(igsf2) &&
        gsft.get<tid>(igsf1) == tables.gsf_table.get<tid>(igsf2)) {
      dist = 0.001;
    }
  }
  return dist;
}
