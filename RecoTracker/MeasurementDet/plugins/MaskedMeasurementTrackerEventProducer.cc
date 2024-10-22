#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

class dso_hidden MaskedMeasurementTrackerEventProducer final : public edm::stream::EDProducer<> {
public:
  explicit MaskedMeasurementTrackerEventProducer(const edm::ParameterSet &iConfig);
  ~MaskedMeasurementTrackerEventProducer() override {}

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  typedef edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > StripMask;
  typedef edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > PixelMask;
  typedef edm::ContainerMask<edmNew::DetSetVector<Phase2TrackerCluster1D> > Phase2OTMask;

  edm::EDGetTokenT<MeasurementTrackerEvent> src_;

  bool skipClusters_;
  bool phase2skipClusters_;

  edm::EDGetTokenT<StripMask> maskStrips_;
  edm::EDGetTokenT<PixelMask> maskPixels_;
  edm::EDGetTokenT<Phase2OTMask> maskPhase2OTs_;
};

MaskedMeasurementTrackerEventProducer::MaskedMeasurementTrackerEventProducer(const edm::ParameterSet &iConfig)
    : src_(consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>("src"))),
      skipClusters_(false),
      phase2skipClusters_(false) {
  //FIXME:temporary solution in order to use this class for both phase0/1 and phase2
  if (iConfig.existsAs<edm::InputTag>("clustersToSkip")) {
    skipClusters_ = true;
    edm::InputTag clustersToSkip = iConfig.getParameter<edm::InputTag>("clustersToSkip");
    maskPixels_ = consumes<PixelMask>(clustersToSkip);
    maskStrips_ = consumes<StripMask>(clustersToSkip);
  }
  if (iConfig.existsAs<edm::InputTag>("phase2clustersToSkip")) {
    phase2skipClusters_ = true;
    edm::InputTag phase2clustersToSkip = iConfig.getParameter<edm::InputTag>("phase2clustersToSkip");
    maskPixels_ = consumes<PixelMask>(phase2clustersToSkip);
    maskPhase2OTs_ = consumes<Phase2OTMask>(phase2clustersToSkip);
  }
  produces<MeasurementTrackerEvent>();
}

void MaskedMeasurementTrackerEventProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<MeasurementTrackerEvent> mte;
  iEvent.getByToken(src_, mte);

  // prepare output
  std::unique_ptr<MeasurementTrackerEvent> out;

  if (skipClusters_) {
    edm::Handle<PixelMask> maskPixels;
    iEvent.getByToken(maskPixels_, maskPixels);
    edm::Handle<StripMask> maskStrips;
    iEvent.getByToken(maskStrips_, maskStrips);

    out = std::make_unique<MeasurementTrackerEvent>(*mte, *maskStrips, *maskPixels);

  } else if (phase2skipClusters_) {
    edm::Handle<PixelMask> maskPixels;
    iEvent.getByToken(maskPixels_, maskPixels);
    edm::Handle<Phase2OTMask> maskPhase2OTs;
    iEvent.getByToken(maskPhase2OTs_, maskPhase2OTs);

    out = std::make_unique<MeasurementTrackerEvent>(*mte, *maskPixels, *maskPhase2OTs);
  }

  // put into event
  iEvent.put(std::move(out));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MaskedMeasurementTrackerEventProducer);
