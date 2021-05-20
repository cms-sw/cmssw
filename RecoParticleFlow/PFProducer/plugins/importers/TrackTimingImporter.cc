#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackAlgoTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// this doesn't actually import anything,
// but rather applies time stamps to tracks after they are all inserted

class TrackTimingImporter : public BlockElementImporterBase {
public:
  TrackTimingImporter(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
      : BlockElementImporterBase(conf, cc),
        useTimeQuality_(conf.existsAs<edm::InputTag>("timeQualityMap")),
        timeQualityThreshold_(useTimeQuality_ ? conf.getParameter<double>("timeQualityThreshold") : -99.),
        srcTime_(cc.consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("timeValueMap"))),
        srcTimeError_(cc.consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("timeErrorMap"))),
        srcTimeQuality_(useTimeQuality_
                            ? cc.consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("timeQualityMap"))
                            : edm::EDGetTokenT<edm::ValueMap<float>>()),
        srcTimeGsf_(cc.consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("timeValueMapGsf"))),
        srcTimeErrorGsf_(cc.consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("timeErrorMapGsf"))),
        srcTimeQualityGsf_(
            useTimeQuality_ ? cc.consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("timeQualityMapGsf"))
                            : edm::EDGetTokenT<edm::ValueMap<float>>()),
        debug_(conf.getUntrackedParameter<bool>("debug", false)) {}

  void importToBlock(const edm::Event&, ElementList&) const override;

private:
  const bool useTimeQuality_;
  const double timeQualityThreshold_;
  edm::EDGetTokenT<edm::ValueMap<float>> srcTime_, srcTimeError_, srcTimeQuality_, srcTimeGsf_, srcTimeErrorGsf_,
      srcTimeQualityGsf_;
  const bool debug_;
};

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, TrackTimingImporter, "TrackTimingImporter");

void TrackTimingImporter::importToBlock(const edm::Event& e, BlockElementImporterBase::ElementList& elems) const {
  typedef BlockElementImporterBase::ElementList::value_type ElementType;

  auto const& time = e.get(srcTime_);
  auto const& timeErr = e.get(srcTimeError_);
  auto const& timeGsf = e.get(srcTimeGsf_);
  auto const& timeErrGsf = e.get(srcTimeErrorGsf_);

  edm::Handle<edm::ValueMap<float>> timeQualH, timeQualGsfH;
  if (useTimeQuality_) {
    e.getByToken(srcTimeQuality_, timeQualH);
    e.getByToken(srcTimeQualityGsf_, timeQualGsfH);
  }

  for (auto& elem : elems) {
    if (reco::PFBlockElement::TRACK == elem->type()) {
      const auto& ref = elem->trackRef();
      if (time.contains(ref.id())) {
        const bool assocQuality = useTimeQuality_ ? (*timeQualH)[ref] > timeQualityThreshold_ : true;
        if (assocQuality) {
          elem->setTime(time[ref], timeErr[ref]);
        } else {
          elem->setTime(0., -1.);
        }
      }
      if (debug_) {
        edm::LogInfo("TrackTimingImporter")
            << "Track with pT / eta " << ref->pt() << " / " << ref->eta() << " has time: " << elem->time() << " +/- "
            << elem->timeError() << std::endl;
      }
    } else if (reco::PFBlockElement::GSF == elem->type()) {
      const auto& ref = static_cast<const reco::PFBlockElementGsfTrack*>(elem.get())->GsftrackRef();
      if (timeGsf.contains(ref.id())) {
        const bool assocQuality = useTimeQuality_ ? (*timeQualGsfH)[ref] > timeQualityThreshold_ : true;
        if (assocQuality) {
          elem->setTime(timeGsf[ref], timeErrGsf[ref]);
        } else {
          elem->setTime(0., -1.);
        }
      }
      if (debug_) {
        edm::LogInfo("TrackTimingImporter")
            << "Track with pT / eta " << ref->pt() << " / " << ref->eta() << " has time: " << elem->time() << " +/- "
            << elem->timeError() << std::endl;
      }
    }
  }
}
