#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

class dso_hidden MaskedMeasurementTrackerEventProducer final : public edm::stream::EDProducer<> {
public:
  explicit MaskedMeasurementTrackerEventProducer(const edm::ParameterSet &iConfig);
  ~MaskedMeasurementTrackerEventProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  typedef edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > StripMask;
  typedef edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > PixelMask;
  typedef edm::ContainerMask<edmNew::DetSetVector<Phase2TrackerCluster1D> > Phase2OTMask;

  const edm::EDGetTokenT<MeasurementTrackerEvent> src_;

  const bool skipClusters_;
  const bool phase2skipClusters_;

  edm::EDGetTokenT<StripMask> maskStrips_;
  edm::EDGetTokenT<PixelMask> maskPixels_;
  edm::EDGetTokenT<Phase2OTMask> maskPhase2OTs_;
};

void MaskedMeasurementTrackerEventProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("MeasurementTrackerEvent"));
  desc.add<edm::InputTag>("clustersToSkip", edm::InputTag(""))->setComment("keep empty string for Phase2");
  desc.add<edm::InputTag>("phase2clustersToSkip", edm::InputTag(""))->setComment("keep empty string for Phase1");
  descriptions.addWithDefaultLabel(desc);
}

MaskedMeasurementTrackerEventProducer::MaskedMeasurementTrackerEventProducer(const edm::ParameterSet &iConfig)
    : src_(consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>("src"))),
      skipClusters_(!iConfig.getParameter<edm::InputTag>("clustersToSkip").label().empty()),
      phase2skipClusters_(!iConfig.getParameter<edm::InputTag>("phase2clustersToSkip").label().empty()) {
  if (skipClusters_) {
    edm::InputTag clustersToSkip = iConfig.getParameter<edm::InputTag>("clustersToSkip");
    maskPixels_ = consumes<PixelMask>(clustersToSkip);
    maskStrips_ = consumes<StripMask>(clustersToSkip);
  }

  if (phase2skipClusters_) {
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
