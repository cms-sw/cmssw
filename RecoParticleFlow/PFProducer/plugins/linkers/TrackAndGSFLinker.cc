#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

using namespace edm::soa::col;

class TrackAndGSFLinker : public BlockElementLinkerBase {
public:
  TrackAndGSFLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        useKDTree_(conf.getParameter<bool>("useKDTree")),
        useConvertedBrems_(conf.getParameter<bool>("useConvertedBrems")),
        debug_(conf.getUntrackedParameter<bool>("debug", false)) {}

  double testLink(size_t ielem1,
                  size_t ielem2,
                  reco::PFBlockElement::Type type1,
                  reco::PFBlockElement::Type type2,
                  const ElementListConst& elements,
                  const PFTables& tables,
                  const reco::PFMultiLinksIndex& multilinks) const override;

private:
  bool useKDTree_, useConvertedBrems_, debug_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, TrackAndGSFLinker, "TrackAndGSFLinker");

double TrackAndGSFLinker::testLink(size_t ielem1,
                                   size_t ielem2,
                                   reco::PFBlockElement::Type type1,
                                   reco::PFBlockElement::Type type2,
                                   const ElementListConst& elements,
                                   const PFTables& tables,
                                   const reco::PFMultiLinksIndex& multilinks) const {
  double dist = -1.0;
  const auto& ttv = tables.track_table_vertex;

  size_t itrack_elem = 0;
  size_t igsf_elem = 0;
  if (type1 < type2) {
    itrack_elem = ielem1;
    igsf_elem = ielem2;
  } else {
    itrack_elem = ielem2;
    igsf_elem = ielem1;
  }
  const size_t itrack = tables.element_to_track[itrack_elem];
  const size_t igsf = tables.element_to_gsf[igsf_elem];

  const auto kf_nn = ttv.get<pf::track::KfTrackRefIsNonNull>(itrack);
  const auto kf_key = ttv.get<pf::track::KfTrackRefKey>(itrack);
  const auto kf_base_key = ttv.get<pf::track::KfTrackRefBaseKey>(itrack);

  if (tables.gsf_table.get<pf::track::KfPFRecTrackRefIsNonNull>(igsf)) {
    const auto gsf_nn = tables.gsf_table.get<pf::track::KfTrackRefIsNonNull>(igsf);
    const auto gsf_key = tables.gsf_table.get<pf::track::KfTrackRefKey>(igsf);

    if (gsf_nn && kf_nn && kf_key == gsf_key) {
      dist = 0.001;
    }
  }

  //override for converted brems
  if (useConvertedBrems_) {
    if (ttv.get<pf::track::IsLinkedToDisplacedVertex>(itrack)) {
      for (size_t iconvbrem : tables.gsf_to_convbrem[igsf]) {
        if (ttv.get<pf::track::TrackType_FROM_GAMMACONV>(itrack) &&
            kf_key == tables.gsf_convbrem_table.get<pf::track::ConvBremRefKey>(iconvbrem)) {
          dist = 0.001;
        } else {  // check the base ref as well (for dedicated conversions?)
          if (tables.gsf_convbrem_table.get<pf::track::ConvBremRefBaseKey>(iconvbrem) == kf_base_key) {
            dist = 0.001;
          }
        }
      }
    }
  }
  return dist;
}
