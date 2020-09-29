#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

using namespace edm::soa::col;

class HCALAndBREMLinker : public BlockElementLinkerBase {
public:
  HCALAndBREMLinker(const edm::ParameterSet& conf)
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

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, HCALAndBREMLinker, "HCALAndBREMLinker");

double HCALAndBREMLinker::testLink(size_t ielem1,
                                   size_t ielem2,
                                   reco::PFBlockElement::Type type1,
                                   reco::PFBlockElement::Type type2,
                                   const ElementListConst& elements,
                                   const PFTables& tables,
                                   const reco::PFMultiLinksIndex& multilinks) const {
  double dist(-1.0);
  size_t ihcal_elem;
  size_t ibrem_elem;
  if (type1 < type2) {
    ihcal_elem = ielem1;
    ibrem_elem = ielem2;
  } else {
    ihcal_elem = ielem2;
    ibrem_elem = ielem1;
  }
  const size_t ihcal = tables.clusters_hcal.element_to_cluster[ihcal_elem];
  const size_t ibrem = tables.element_to_brem[ibrem_elem];

  if (tables.brem_table_hcalent.get<pf::track::ExtrapolationValid>(ibrem)) {
    const auto& rechits = tables.clusters_hcal.cluster_to_rechit.at(ihcal);
    dist = LinkByRecHit::testTrackAndClusterByRecHit(ihcal,
                                                     rechits,
                                                     tables.clusters_hcal.cluster_table,
                                                     tables.clusters_hcal.rechit_table,
                                                     ibrem,
                                                     tables.brem_table,                //NOT USED
                                                     tables.brem_table_ecalshowermax,  //NOT USED
                                                     tables.brem_table_hcalent,
                                                     tables.track_table_hcalex,  //NOT USED
                                                     tables.track_table_ho,      //NOT USED
                                                     true);

    //According to testTrackAndClusterByRecHit, HCAL and BREM linker ALWAYS gives dist=-1
    //Therefore, likely we can completely throw out all this code
    assert(dist == -1.0);
  }
  return dist;
}
