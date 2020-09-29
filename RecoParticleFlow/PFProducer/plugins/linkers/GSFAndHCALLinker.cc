#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class GSFAndHCALLinker : public BlockElementLinkerBase {
public:
  GSFAndHCALLinker(const edm::ParameterSet& conf)
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

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, GSFAndHCALLinker, "GSFAndHCALLinker");

double GSFAndHCALLinker::testLink(size_t ielem1,
                                  size_t ielem2,
                                  reco::PFBlockElement::Type type1,
                                  reco::PFBlockElement::Type type2,
                                  const ElementListConst& elements,
                                  const PFTables& tables,
                                  const reco::PFMultiLinksIndex& multilinks) const {
  using namespace edm::soa::col;

  size_t ihcal_elem = 0;
  size_t igsf_elem = 0;

  double dist(-1.0);

  if (type1 < type2) {
    ihcal_elem = ielem1;
    igsf_elem = ielem2;
  } else {
    ihcal_elem = ielem2;
    igsf_elem = ielem1;
  }

  const size_t ihcal = tables.clusters_hcal.element_to_cluster[ihcal_elem];
  const size_t igsf = tables.element_to_gsf[igsf_elem];

  if (tables.gsf_table_hcalent.get<pf::track::ExtrapolationValid>(igsf)) {
    dist = LinkByRecHit::testTrackAndClusterByRecHit(ihcal,
                                                     tables.clusters_hcal.cluster_to_rechit.at(ihcal),
                                                     tables.clusters_hcal.cluster_table,
                                                     tables.clusters_hcal.rechit_table,
                                                     igsf,
                                                     tables.gsf_table,
                                                     tables.gsf_table_ecalshowermax,
                                                     tables.gsf_table_hcalent,
                                                     tables.gsf_table_hcalex,
                                                     tables.track_table_ho,  // NOT USED
                                                     false);
  }

  if (debug_) {
    if (dist > 0.) {
      std::cout << " Here a link has been established"
                << " between a GSF track an Hcal with dist  " << dist << std::endl;
    } else {
      if (debug_)
        std::cout << " No link found " << std::endl;
    }
  }

  return dist;
}
