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

  size_t itrack = tables.element_to_track_[itrack_elem];
  size_t iho = tables.clusters_ho_.element_to_cluster_[iho_elem];

  if (tables.track_table_vertex_.get<pf::track::Pt>(itrack) > 3.00001 &&
      tables.track_table_ho_.get<pf::track::ExtrapolationValid>(itrack)) {
    dist = LinkByRecHit::testTrackAndClusterByRecHit(iho,
                                                     tables.clusters_ho_.cluster_to_rechit_.at(iho),
                                                     tables.clusters_ho_.cluster_table_,
                                                     tables.clusters_ho_.rechit_table_,
                                                     itrack,
                                                     tables.track_table_vertex_,
                                                     tables.track_table_ecalshowermax_,
                                                     tables.track_table_hcalent_,
                                                     tables.track_table_hcalex_,
                                                     tables.track_table_ho_,
                                                     false);

  } else {
    dist = -1.;
  }
  return dist;
}
