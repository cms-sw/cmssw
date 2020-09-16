#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

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
  const auto* elem1 = elements[ielem1];
  const auto* elem2 = elements[ielem2];
  constexpr reco::PFBlockElement::TrackType T_FROM_GAMMACONV = reco::PFBlockElement::T_FROM_GAMMACONV;
  double dist = -1.0;
  const reco::PFBlockElementGsfTrack* gsfelem(nullptr);
  const reco::PFBlockElementTrack* tkelem(nullptr);
  if (type1 < type2) {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem1);
    gsfelem = static_cast<const reco::PFBlockElementGsfTrack*>(elem2);
  } else {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem2);
    gsfelem = static_cast<const reco::PFBlockElementGsfTrack*>(elem1);
  }

  const reco::PFRecTrackRef& trackref = tkelem->trackRefPF();
  const reco::GsfPFRecTrackRef& gsfref = gsfelem->GsftrackRefPF();
  const reco::TrackRef& kftrackref = trackref->trackRef();
  const reco::TrackBaseRef kftrackrefbase(kftrackref);
  const reco::PFRecTrackRef& refkf = gsfref->kfPFRecTrackRef();
  if (refkf.isNonnull()) {
    const reco::TrackRef& gsftrackref = refkf->trackRef();
    if (gsftrackref.isNonnull() && kftrackref.isNonnull() && kftrackref == gsftrackref) {
      dist = 0.001;
    }
  }

  //override for converted brems
  if (useConvertedBrems_) {
    if (tkelem->isLinkedToDisplacedVertex()) {
      const std::vector<reco::PFRecTrackRef>& convbrems = gsfref->convBremPFRecTrackRef();
      for (const auto& convbrem : convbrems) {
        if (tkelem->trackType(T_FROM_GAMMACONV) && kftrackref == convbrem->trackRef()) {
          dist = 0.001;
        } else {  // check the base ref as well (for dedicated conversions?)
          const reco::TrackBaseRef convbrembase(convbrem->trackRef());
          if (convbrembase == kftrackrefbase) {
            dist = 0.001;
          }
        }
      }
    }
  }
  return dist;
}
