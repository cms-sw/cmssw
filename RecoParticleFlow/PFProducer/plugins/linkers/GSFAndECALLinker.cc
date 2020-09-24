#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class GSFAndECALLinker : public BlockElementLinkerBase {
public:
  GSFAndECALLinker(const edm::ParameterSet& conf)
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

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, GSFAndECALLinker, "GSFAndECALLinker");

double GSFAndECALLinker::testLink(size_t ielem1,
                                  size_t ielem2,
                                  reco::PFBlockElement::Type type1,
                                  reco::PFBlockElement::Type type2,
                                  const ElementListConst& elements,
                                  const PFTables& tables,
                                  const reco::PFMultiLinksIndex& multilinks) const {
  using namespace edm::soa::col;

  size_t iecal_elem = 0;
  size_t igsf_elem = 0;

  double dist(-1.0);

  if (type1 < type2) {
    iecal_elem = ielem1;
    igsf_elem = ielem2;
  } else {
    iecal_elem = ielem2;
    igsf_elem = ielem1;
  }

  size_t iecal = tables.clusters_ecal_.element_to_cluster_[iecal_elem];
  size_t igsf = tables.element_to_gsf_[igsf_elem];

  if (tables.gsf_table_ecalshowermax_.get<pf::track::ExtrapolationValid>(igsf)) {
    dist = LinkByRecHit::testTrackAndClusterByRecHit(iecal,
                                                     tables.clusters_ecal_.cluster_to_rechit_.at(iecal),
                                                     tables.clusters_ecal_.cluster_table_,
                                                     tables.clusters_ecal_.rechit_table_,
                                                     igsf,
                                                     tables.gsf_table_,
                                                     tables.gsf_table_ecalshowermax_,
                                                     tables.gsf_table_hcalent_,
                                                     tables.gsf_table_hcalex_,
                                                     tables.track_table_ho_,  // NOT USED
                                                     false);
  }

  if (debug_) {
    if (dist > 0.) {
      std::cout << " Here a link has been established"
                << " between a GSF track an Ecal with dist  " << dist << std::endl;
    } else {
      if (debug_)
        std::cout << " No link found " << std::endl;
    }
  }

  return dist;
}
