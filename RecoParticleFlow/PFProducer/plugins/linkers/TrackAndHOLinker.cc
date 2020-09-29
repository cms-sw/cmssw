#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

using namespace edm::soa::col;

class TrackAndHOLinker : public BlockElementLinkerBase {
public:
  TrackAndHOLinker(const edm::ParameterSet& conf)
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

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, TrackAndHOLinker, "TrackAndHOLinker");

double TrackAndHOLinker::testLink(size_t ielem1,
                                  size_t ielem2,
                                  reco::PFBlockElement::Type type1,
                                  reco::PFBlockElement::Type type2,
                                  const ElementListConst& elements,
                                  const PFTables& tables,
                                  const reco::PFMultiLinksIndex& multilinks) const {
  size_t iho_elem = 0;
  size_t itrack_elem = 0;

  double dist(-1.0);
  if (type1 < type2) {
    itrack_elem = ielem1;
    iho_elem = ielem2;
  } else {
    itrack_elem = ielem2;
    iho_elem = ielem1;
  }

  const size_t itrack = tables.element_to_track[itrack_elem];
  const size_t iho = tables.clusters_ho.element_to_cluster[iho_elem];

  if (tables.track_table_vertex.get<pf::track::Pt>(itrack) > 3.00001 &&
      tables.track_table_ho.get<pf::track::ExtrapolationValid>(itrack)) {
    dist = LinkByRecHit::testTrackAndClusterByRecHit(iho,
                                                     tables.clusters_ho.cluster_to_rechit.at(iho),
                                                     tables.clusters_ho.cluster_table,
                                                     tables.clusters_ho.rechit_table,
                                                     itrack,
                                                     tables.track_table_vertex,
                                                     tables.track_table_ecalshowermax,
                                                     tables.track_table_hcalent,
                                                     tables.track_table_hcalex,
                                                     tables.track_table_ho,
                                                     false);

  } else {
    dist = -1.;
  }
  return dist;
}
