#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

using namespace edm::soa::col;

class TrackAndTrackLinker : public BlockElementLinkerBase {
public:
  TrackAndTrackLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        useKDTree_(conf.getParameter<bool>("useKDTree")),
        debug_(conf.getUntrackedParameter<bool>("debug", false)) {}

  bool linkPrefilter(size_t ielem1,
                     size_t ielem2,
                     reco::PFBlockElement::Type type1,
                     reco::PFBlockElement::Type type2,
                     const PFTables& tables,
                     const reco::PFMultiLinksIndex& multilinks,
                     const reco::PFBlockElement*,
                     const reco::PFBlockElement*) const override;

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

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, TrackAndTrackLinker, "TrackAndTrackLinker");

bool TrackAndTrackLinker::linkPrefilter(size_t ielem1,
                                        size_t ielem2,
                                        reco::PFBlockElement::Type type1,
                                        reco::PFBlockElement::Type type2,
                                        const PFTables& tables,
                                        const reco::PFMultiLinksIndex& multilinks,
                                        const reco::PFBlockElement* e1,
                                        const reco::PFBlockElement* e2) const {
  size_t ielem1_track = tables.element_to_track_[ielem1];
  size_t ielem2_track = tables.element_to_track_[ielem2];
  return (tables.track_table_vertex_.get<pf::track::IsLinkedToDisplacedVertex>(ielem1_track) ||
          tables.track_table_vertex_.get<pf::track::IsLinkedToDisplacedVertex>(ielem2_track));
}

double TrackAndTrackLinker::testLink(size_t ielem1,
                                     size_t ielem2,
                                     reco::PFBlockElement::Type type1,
                                     reco::PFBlockElement::Type type2,
                                     const ElementListConst& elements,
                                     const PFTables& tables,
                                     const reco::PFMultiLinksIndex& multilinks) const {
  size_t ielem1_track = tables.element_to_track_[ielem1];
  size_t ielem2_track = tables.element_to_track_[ielem2];

  double dist = -1.0;

  const bool dv1_to_nn = tables.track_table_vertex_.get<pf::track::DisplacedVertexRef_TO_DISP_IsNonNull>(ielem1_track);
  const bool dv2_to_nn = tables.track_table_vertex_.get<pf::track::DisplacedVertexRef_TO_DISP_IsNonNull>(ielem2_track);
  const bool dv1_from_nn =
      tables.track_table_vertex_.get<pf::track::DisplacedVertexRef_FROM_DISP_IsNonNull>(ielem1_track);
  const bool dv2_from_nn =
      tables.track_table_vertex_.get<pf::track::DisplacedVertexRef_FROM_DISP_IsNonNull>(ielem2_track);

  const auto& dv1_to_key = tables.track_table_vertex_.get<pf::track::DisplacedVertexRef_TO_DISP_Key>(ielem1_track);
  const auto& dv2_to_key = tables.track_table_vertex_.get<pf::track::DisplacedVertexRef_TO_DISP_Key>(ielem2_track);
  const auto& dv1_from_key = tables.track_table_vertex_.get<pf::track::DisplacedVertexRef_FROM_DISP_Key>(ielem1_track);
  const auto& dv2_from_key = tables.track_table_vertex_.get<pf::track::DisplacedVertexRef_FROM_DISP_Key>(ielem2_track);

  if (dv1_to_nn && dv2_from_nn) {
    if (dv1_to_key == dv2_from_key) {
      dist = 1.0;
    }
  }

  if (dv1_from_nn && dv2_to_nn) {
    if (dv1_from_key == dv2_to_key) {
      dist = 1.0;
    }
  }

  if (dv1_from_nn && dv2_from_nn) {
    if (dv1_from_key == dv2_from_key) {
      dist = 1.0;
    }
  }

  if (tables.track_table_vertex_.get<pf::track::TrackType_FROM_GAMMACONV>(ielem1_track) &&
      tables.track_table_vertex_.get<pf::track::TrackType_FROM_GAMMACONV>(ielem2_track)) {
    const auto& convrefs1 = tables.track_to_convrefs_[ielem1_track];
    const auto& convrefs2 = tables.track_to_convrefs_[ielem2_track];

    for (size_t convref1 : convrefs1) {
      for (size_t convref2 : convrefs2) {
        const bool cr1_nn = tables.convref_table_.get<pf::track::ConvRefIsNonNull>(convref1);
        const bool cr2_nn = tables.convref_table_.get<pf::track::ConvRefIsNonNull>(convref2);
        const auto& cr1_key = tables.convref_table_.get<pf::track::ConvRefKey>(convref1);
        const auto& cr2_key = tables.convref_table_.get<pf::track::ConvRefKey>(convref2);

        if (cr1_nn && cr2_nn && cr1_key == cr2_key) {
          dist = 1.0;
          break;
        }
      }
    }
  }

  const bool tt1_from_v0 = tables.track_table_vertex_.get<pf::track::TrackType_FROM_V0>(ielem1_track);
  const bool tt2_from_v0 = tables.track_table_vertex_.get<pf::track::TrackType_FROM_V0>(ielem2_track);
  const bool tt1_v0_nn = tables.track_table_vertex_.get<pf::track::V0RefIsNonNull>(ielem1_track);
  const bool tt2_v0_nn = tables.track_table_vertex_.get<pf::track::V0RefIsNonNull>(ielem2_track);

  const auto& tt1_v0_key = tables.track_table_vertex_.get<pf::track::V0RefKey>(ielem1_track);
  const auto& tt2_v0_key = tables.track_table_vertex_.get<pf::track::V0RefKey>(ielem2_track);

  if (tt1_from_v0 && tt2_from_v0) {
    if (tt1_v0_nn && tt2_v0_nn) {
      if (tt1_v0_key == tt2_v0_key) {
        dist = 1.0;
      }
    }
  }

  return dist;
}
