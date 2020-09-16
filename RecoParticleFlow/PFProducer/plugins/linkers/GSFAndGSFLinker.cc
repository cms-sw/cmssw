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
  using namespace edm::soa::col;

  double dist = -1.0;

  size_t igsf1 = tables.element_to_gsf_[ielem1];
  size_t igsf2 = tables.element_to_gsf_[ielem2];

  if (tables.gsf_table_.get<pf::track::GsfTrackRefPFIsNonNull>(igsf1) &&
      tables.gsf_table_.get<pf::track::GsfTrackRefPFIsNonNull>(igsf2)) {
    if (tables.gsf_table_.get<pf::track::TrackType_FROM_GAMMACONV>(igsf1) !=  // we want **one** primary GSF
            tables.gsf_table_.get<pf::track::TrackType_FROM_GAMMACONV>(igsf2) &&
        tables.gsf_table_.get<pf::track::GsfTrackRefPFTrackId>(igsf1) ==
            tables.gsf_table_.get<pf::track::GsfTrackRefPFTrackId>(igsf2)) {
      dist = 0.001;
    }
  }
  return dist;
}
