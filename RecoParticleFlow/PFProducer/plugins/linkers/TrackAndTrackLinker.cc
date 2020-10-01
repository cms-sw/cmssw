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
                     const reco::PFMultiLinksIndex& multilinks) const override;

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
                                        const reco::PFMultiLinksIndex& multilinks) const {
  using IsLinkedToDisplacedVertex = pf::track::IsLinkedToDisplacedVertex;

  const size_t ielem1_track = tables.element_to_track[ielem1];
  const size_t ielem2_track = tables.element_to_track[ielem2];
  const auto& ttv = tables.track_table_vertex;

  return (ttv.get<IsLinkedToDisplacedVertex>(ielem1_track) || ttv.get<IsLinkedToDisplacedVertex>(ielem2_track));
}

double TrackAndTrackLinker::testLink(size_t ielem1,
                                     size_t ielem2,
                                     reco::PFBlockElement::Type type1,
                                     reco::PFBlockElement::Type type2,
                                     const ElementListConst& elements,
                                     const PFTables& tables,
                                     const reco::PFMultiLinksIndex& multilinks) const {
  size_t ielem1_track = tables.element_to_track[ielem1];
  size_t ielem2_track = tables.element_to_track[ielem2];
  const auto& ttv = tables.track_table_vertex;
  const auto& crt = tables.convref_table;

  double dist = -1.0;

  using TO_DISP_IsNonNull = pf::track::DisplacedVertexRef_TO_DISP_IsNonNull;
  using FROM_DISP_IsNonNull = pf::track::DisplacedVertexRef_FROM_DISP_IsNonNull;
  using TO_DISP_Key = pf::track::DisplacedVertexRef_TO_DISP_Key;
  using FROM_DISP_Key = pf::track::DisplacedVertexRef_FROM_DISP_Key;

  const bool dv1_to_nn = ttv.get<TO_DISP_IsNonNull>(ielem1_track);
  const bool dv2_to_nn = ttv.get<TO_DISP_IsNonNull>(ielem2_track);
  const bool dv1_from_nn = ttv.get<FROM_DISP_IsNonNull>(ielem1_track);
  const bool dv2_from_nn = ttv.get<FROM_DISP_IsNonNull>(ielem2_track);

  const auto& dv1_to_key = ttv.get<TO_DISP_Key>(ielem1_track);
  const auto& dv2_to_key = ttv.get<TO_DISP_Key>(ielem2_track);
  const auto& dv1_from_key = ttv.get<FROM_DISP_Key>(ielem1_track);
  const auto& dv2_from_key = ttv.get<FROM_DISP_Key>(ielem2_track);

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

  if (ttv.get<pf::track::TrackType_FROM_GAMMACONV>(ielem1_track) &&
      ttv.get<pf::track::TrackType_FROM_GAMMACONV>(ielem2_track)) {
    const auto& convrefs1 = tables.track_to_convrefs[ielem1_track];
    const auto& convrefs2 = tables.track_to_convrefs[ielem2_track];

    for (size_t convref1 : convrefs1) {
      for (size_t convref2 : convrefs2) {
        const bool cr1_nn = crt.get<pf::track::ConvRefIsNonNull>(convref1);
        const bool cr2_nn = crt.get<pf::track::ConvRefIsNonNull>(convref2);
        const auto& cr1_key = crt.get<pf::track::ConvRefKey>(convref1);
        const auto& cr2_key = crt.get<pf::track::ConvRefKey>(convref2);

        if (cr1_nn && cr2_nn && cr1_key == cr2_key) {
          dist = 1.0;
          break;
        }
      }
    }
  }

  const bool tt1_from_v0 = ttv.get<pf::track::TrackType_FROM_V0>(ielem1_track);
  const bool tt2_from_v0 = ttv.get<pf::track::TrackType_FROM_V0>(ielem2_track);
  const bool tt1_v0_nn = ttv.get<pf::track::V0RefIsNonNull>(ielem1_track);
  const bool tt2_v0_nn = ttv.get<pf::track::V0RefIsNonNull>(ielem2_track);

  const auto& tt1_v0_key = ttv.get<pf::track::V0RefKey>(ielem1_track);
  const auto& tt2_v0_key = ttv.get<pf::track::V0RefKey>(ielem2_track);

  if (tt1_from_v0 && tt2_from_v0) {
    if (tt1_v0_nn && tt2_v0_nn) {
      if (tt1_v0_key == tt2_v0_key) {
        dist = 1.0;
      }
    }
  }

  return dist;
}
