#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackAlgoTools.h"

// this doesn't actually import anything,
// but rather applies time stamps to tracks after they are all inserted

class TrackTimingImporter : public BlockElementImporterBase {
public:
  TrackTimingImporter(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes)
      : BlockElementImporterBase(conf, sumes),
        srcTime_(sumes.consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("timeValueMap"))),
        srcTimeError_(sumes.consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("timeErrorMap"))),
        srcTimeGsf_(sumes.consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("timeValueMapGsf"))),
        srcTimeErrorGsf_(sumes.consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("timeErrorMapGsf"))),
        debug_(conf.getUntrackedParameter<bool>("debug", false)) {}

  void importToBlock(const edm::Event&, ElementList&) const override;

private:
  edm::EDGetTokenT<edm::ValueMap<float> > srcTime_, srcTimeError_, srcTimeGsf_, srcTimeErrorGsf_;
  const bool debug_;
};

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, TrackTimingImporter, "TrackTimingImporter");

void TrackTimingImporter::importToBlock(const edm::Event& e, BlockElementImporterBase::ElementList& elems) const {
  typedef BlockElementImporterBase::ElementList::value_type ElementType;

  auto const& time = e.get(srcTime_);
  auto const& timeErr = e.get(srcTimeError_);
  auto const& timeGsf = e.get(srcTimeGsf_);
  auto const& timeErrGsf = e.get(srcTimeErrorGsf_);

  for (auto& elem : elems) {
    if (reco::PFBlockElement::TRACK == elem->type()) {
      const auto& ref = elem->trackRef();
      if (time.contains(ref.id())) {
        elem->setTime(time[ref], timeErr[ref]);
      }
      if (debug_) {
        edm::LogInfo("TrackTimingImporter")
            << "Track with pT / eta " << ref->pt() << " / " << ref->eta() << " has time: " << elem->time() << " +/- "
            << elem->timeError() << std::endl;
      }
    } else if (reco::PFBlockElement::GSF == elem->type()) {
      const auto& ref = static_cast<const reco::PFBlockElementGsfTrack*>(elem.get())->GsftrackRef();
      if (timeGsf.contains(ref.id())) {
        elem->setTime(timeGsf[ref], timeErrGsf[ref]);
      }
      if (debug_) {
        edm::LogInfo("TrackTimingImporter")
            << "Track with pT / eta " << ref->pt() << " / " << ref->eta() << " has time: " << elem->time() << " +/- "
            << elem->timeError() << std::endl;
      }
    }
  }
}
