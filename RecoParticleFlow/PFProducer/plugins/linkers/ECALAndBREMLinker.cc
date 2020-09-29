#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

using namespace edm::soa::col;

class ECALAndBREMLinker : public BlockElementLinkerBase {
public:
  ECALAndBREMLinker(const edm::ParameterSet& conf)
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

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, ECALAndBREMLinker, "ECALAndBREMLinker");

double ECALAndBREMLinker::testLink(size_t ielem1,
                                   size_t ielem2,
                                   reco::PFBlockElement::Type type1,
                                   reco::PFBlockElement::Type type2,
                                   const ElementListConst& elements,
                                   const PFTables& tables,
                                   const reco::PFMultiLinksIndex& multilinks) const {
  double dist(-1.0);

  size_t iecal_elem;
  size_t ibrem_elem;

  if (type1 < type2) {
    iecal_elem = ielem1;
    ibrem_elem = ielem2;
  } else {
    iecal_elem = ielem2;
    ibrem_elem = ielem1;
  }
  const size_t iecal = tables.clusters_ecal.element_to_cluster[iecal_elem];
  const size_t ibrem = tables.element_to_brem[ibrem_elem];

  if (tables.brem_table_ecalshowermax.get<pf::track::ExtrapolationValid>(ibrem)) {
    const auto& rechits = tables.clusters_ecal.cluster_to_rechit.at(iecal);

    //note that the function testTrackAndClusterByRecHit has not been refactored and currently needs inputs
    //also for the track extrapolations that are not used, we pass placeholders for those.
    dist = LinkByRecHit::testTrackAndClusterByRecHit(iecal,
                                                     rechits,
                                                     tables.clusters_ecal.cluster_table,
                                                     tables.clusters_ecal.rechit_table,
                                                     ibrem,
                                                     tables.brem_table,  //NOT USED
                                                     tables.brem_table_ecalshowermax,
                                                     tables.brem_table_hcalent,  //NOT USED
                                                     tables.track_table_hcalex,  //NOT USED
                                                     tables.track_table_ho,      //NOT USED
                                                     true);
  }
  return dist;
}
