#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/BeamSpot/interface/alpaka/BeamSpotDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/Records/interface/PixelCPEFastParamsRecord.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/alpaka/PixelCPEFastParamsCollection.h"

#include "PixelRecHitKernel.h"

#include <optional>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelRecHitExtendedAlpaka : public global::EDProducer<> {
  public:
    explicit SiPixelRecHitExtendedAlpaka(const edm::ParameterSet& iConfig);
    ~SiPixelRecHitExtendedAlpaka() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    const device::EDGetToken<reco::TrackingRecHitsSoACollection> pixelRecHitToken_;
    const device::EDGetToken<reco::TrackingRecHitsSoACollection> trackerRecHitToken_;

    const device::EDPutToken<reco::TrackingRecHitsSoACollection> outputRecHitsSoAToken_;
  };

  SiPixelRecHitExtendedAlpaka::SiPixelRecHitExtendedAlpaka(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        pixelRecHitToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitsSoA"))),
        trackerRecHitToken_(consumes(iConfig.getParameter<edm::InputTag>("trackerRecHitsSoA"))),
        outputRecHitsSoAToken_(produces()) {}

  void SiPixelRecHitExtendedAlpaka::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("pixelRecHitsSoA", edm::InputTag("hltPhase2SiPixelRecHitsSoA"));
    desc.add<edm::InputTag>("trackerRecHitsSoA", edm::InputTag("phase2OTRecHitsSoAConverter"));

    descriptions.addWithDefaultLabel(desc);
  }

  void SiPixelRecHitExtendedAlpaka::produce(edm::StreamID streamID,
                                            device::Event& iEvent,
                                            const device::EventSetup& es) const {
    // get both Pixel and Tracker recHits
    auto queue = iEvent.queue();
    const auto& pixelRecHitsSoA = iEvent.get(pixelRecHitToken_);
    const auto& otRecHitsSoA = iEvent.get(trackerRecHitToken_);
    std::cout << "----------------- Merging Pixel and Tracker RecHits -----------------" << std::endl;
    const int nPixelHits = pixelRecHitsSoA.nHits();
    std::cout << "Number of Pixel recHits: " << nPixelHits << std::endl;
    const int nTrackerHits = otRecHitsSoA.nHits();
    std::cout << "Number of Tracker recHits: " << nTrackerHits << std::endl;
    const int nTotHits = nPixelHits + nTrackerHits;
    std::cout << "Number of Pixel modules: " << pixelRecHitsSoA.nModules() << std::endl;
    std::cout << "Number of Tracker modules: " << otRecHitsSoA.nModules() << std::endl;
    const int nTotModules = pixelRecHitsSoA.nModules() + otRecHitsSoA.nModules();

    auto outputSoA = reco::TrackingRecHitsSoACollection(queue, nTotHits, nTotModules);
    std::cout << "Total number of recHits: " << outputSoA.nHits() << std::endl;

    // copy all columns from pixelRecHitsSoA and otRecHitsSoA to outputSoA
    // xLocal
    auto xLocalOutputPixel = cms::alpakatools::make_device_view(queue, outputSoA.view().xLocal(), nPixelHits);
    auto xLocalOutputTracker =
        cms::alpakatools::make_device_view(queue, outputSoA.view().xLocal() + nPixelHits, nTrackerHits);
    auto xLocalPixel = cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().xLocal(), nPixelHits);
    auto xLocalTracker = cms::alpakatools::make_device_view(queue, otRecHitsSoA.view().xLocal(), nTrackerHits);
    alpaka::memcpy(queue, xLocalOutputPixel, xLocalPixel);
    alpaka::memcpy(queue, xLocalOutputTracker, xLocalTracker);

    // yLocal
    auto yLocalOutputPixel = cms::alpakatools::make_device_view(queue, outputSoA.view().yLocal(), nPixelHits);
    auto yLocalOutputTracker =
        cms::alpakatools::make_device_view(queue, outputSoA.view().yLocal() + nPixelHits, nTrackerHits);
    auto yLocalPixel = cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().yLocal(), nPixelHits);
    auto yLocalTracker = cms::alpakatools::make_device_view(queue, otRecHitsSoA.view().yLocal(), nTrackerHits);
    alpaka::memcpy(queue, yLocalOutputPixel, yLocalPixel);
    alpaka::memcpy(queue, yLocalOutputTracker, yLocalTracker);

    // xerrLocal
    auto xerrLocalOutputPixel = cms::alpakatools::make_device_view(queue, outputSoA.view().xerrLocal(), nPixelHits);
    auto xerrLocalOutputTracker =
        cms::alpakatools::make_device_view(queue, outputSoA.view().xerrLocal() + nPixelHits, nTrackerHits);
    auto xerrLocalPixel = cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().xerrLocal(), nPixelHits);
    auto xerrLocalTracker = cms::alpakatools::make_device_view(queue, otRecHitsSoA.view().xerrLocal(), nTrackerHits);
    alpaka::memcpy(queue, xerrLocalOutputPixel, xerrLocalPixel);
    alpaka::memcpy(queue, xerrLocalOutputTracker, xerrLocalTracker);

    // yerrLocal
    auto yerrLocalOutputPixel = cms::alpakatools::make_device_view(queue, outputSoA.view().yerrLocal(), nPixelHits);
    auto yerrLocalOutputTracker =
        cms::alpakatools::make_device_view(queue, outputSoA.view().yerrLocal() + nPixelHits, nTrackerHits);
    auto yerrLocalPixel = cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().yerrLocal(), nPixelHits);
    auto yerrLocalTracker = cms::alpakatools::make_device_view(queue, otRecHitsSoA.view().yerrLocal(), nTrackerHits);
    alpaka::memcpy(queue, yerrLocalOutputPixel, yerrLocalPixel);
    alpaka::memcpy(queue, yerrLocalOutputTracker, yerrLocalTracker);

    // xGlobal
    auto xGlobalOutputPixel = cms::alpakatools::make_device_view(queue, outputSoA.view().xGlobal(), nPixelHits);
    auto xGlobalOutputTracker =
        cms::alpakatools::make_device_view(queue, outputSoA.view().xGlobal() + nPixelHits, nTrackerHits);
    auto xGlobalPixel = cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().xGlobal(), nPixelHits);
    auto xGlobalTracker = cms::alpakatools::make_device_view(queue, otRecHitsSoA.view().xGlobal(), nTrackerHits);
    alpaka::memcpy(queue, xGlobalOutputPixel, xGlobalPixel);
    alpaka::memcpy(queue, xGlobalOutputTracker, xGlobalTracker);

    // yGlobal
    auto yGlobalOutputPixel = cms::alpakatools::make_device_view(queue, outputSoA.view().yGlobal(), nPixelHits);
    auto yGlobalOutputTracker =
        cms::alpakatools::make_device_view(queue, outputSoA.view().yGlobal() + nPixelHits, nTrackerHits);
    auto yGlobalPixel = cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().yGlobal(), nPixelHits);
    auto yGlobalTracker = cms::alpakatools::make_device_view(queue, otRecHitsSoA.view().yGlobal(), nTrackerHits);
    alpaka::memcpy(queue, yGlobalOutputPixel, yGlobalPixel);
    alpaka::memcpy(queue, yGlobalOutputTracker, yGlobalTracker);

    // zGlobal
    auto zGlobalOutputPixel = cms::alpakatools::make_device_view(queue, outputSoA.view().zGlobal(), nPixelHits);
    auto zGlobalOutputTracker =
        cms::alpakatools::make_device_view(queue, outputSoA.view().zGlobal() + nPixelHits, nTrackerHits);
    auto zGlobalPixel = cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().zGlobal(), nPixelHits);
    auto zGlobalTracker = cms::alpakatools::make_device_view(queue, otRecHitsSoA.view().zGlobal(), nTrackerHits);
    alpaka::memcpy(queue, zGlobalOutputPixel, zGlobalPixel);
    alpaka::memcpy(queue, zGlobalOutputTracker, zGlobalTracker);

    // rGlobal
    auto rGlobalOutputPixel = cms::alpakatools::make_device_view(queue, outputSoA.view().rGlobal(), nPixelHits);
    auto rGlobalOutputTracker =
        cms::alpakatools::make_device_view(queue, outputSoA.view().rGlobal() + nPixelHits, nTrackerHits);
    auto rGlobalPixel = cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().rGlobal(), nPixelHits);
    auto rGlobalTracker = cms::alpakatools::make_device_view(queue, otRecHitsSoA.view().rGlobal(), nTrackerHits);
    alpaka::memcpy(queue, rGlobalOutputPixel, rGlobalPixel);
    alpaka::memcpy(queue, rGlobalOutputTracker, rGlobalTracker);

    // iphi
    auto iphiOutputPixel = cms::alpakatools::make_device_view(queue, outputSoA.view().iphi(), nPixelHits);
    auto iphiOutputTracker =
        cms::alpakatools::make_device_view(queue, outputSoA.view().iphi() + nPixelHits, nTrackerHits);
    auto iphiPixel = cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().iphi(), nPixelHits);
    auto iphiTracker = cms::alpakatools::make_device_view(queue, otRecHitsSoA.view().iphi(), nTrackerHits);
    alpaka::memcpy(queue, iphiOutputPixel, iphiPixel);
    alpaka::memcpy(queue, iphiOutputTracker, iphiTracker);

    // chargeAndStatus
    auto chargeAndStatusOutputPixel =
        cms::alpakatools::make_device_view(queue, outputSoA.view().chargeAndStatus(), nPixelHits);
    auto chargeAndStatusOutputTracker =
        cms::alpakatools::make_device_view(queue, outputSoA.view().chargeAndStatus() + nPixelHits, nTrackerHits);
    auto chargeAndStatusPixel =
        cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().chargeAndStatus(), nPixelHits);
    auto chargeAndStatusTracker =
        cms::alpakatools::make_device_view(queue, otRecHitsSoA.view().chargeAndStatus(), nTrackerHits);
    alpaka::memcpy(queue, chargeAndStatusOutputPixel, chargeAndStatusPixel);
    alpaka::memcpy(queue, chargeAndStatusOutputTracker, chargeAndStatusTracker);

    // clusterSizeX
    auto clusterSizeXOutputPixel =
        cms::alpakatools::make_device_view(queue, outputSoA.view().clusterSizeX(), nPixelHits);
    auto clusterSizeXOutputTracker =
        cms::alpakatools::make_device_view(queue, outputSoA.view().clusterSizeX() + nPixelHits, nTrackerHits);
    auto clusterSizeXPixel =
        cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().clusterSizeX(), nPixelHits);
    auto clusterSizeXTracker =
        cms::alpakatools::make_device_view(queue, otRecHitsSoA.view().clusterSizeX(), nTrackerHits);
    alpaka::memcpy(queue, clusterSizeXOutputPixel, clusterSizeXPixel);
    alpaka::memcpy(queue, clusterSizeXOutputTracker, clusterSizeXTracker);

    // clusterSizeY
    auto clusterSizeYOutputPixel =
        cms::alpakatools::make_device_view(queue, outputSoA.view().clusterSizeY(), nPixelHits);
    auto clusterSizeYOutputTracker =
        cms::alpakatools::make_device_view(queue, outputSoA.view().clusterSizeY() + nPixelHits, nTrackerHits);
    auto clusterSizeYPixel =
        cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().clusterSizeY(), nPixelHits);
    auto clusterSizeYTracker =
        cms::alpakatools::make_device_view(queue, otRecHitsSoA.view().clusterSizeY(), nTrackerHits);
    alpaka::memcpy(queue, clusterSizeYOutputPixel, clusterSizeYPixel);
    alpaka::memcpy(queue, clusterSizeYOutputTracker, clusterSizeYTracker);

    // detectorIndex
    auto detectorIndexOutputPixel =
        cms::alpakatools::make_device_view(queue, outputSoA.view().detectorIndex(), nPixelHits);
    auto detectorIndexOutputTracker =
        cms::alpakatools::make_device_view(queue, outputSoA.view().detectorIndex() + nPixelHits, nTrackerHits);
    auto detectorIndexPixel =
        cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().detectorIndex(), nPixelHits);
    auto detectorIndexTracker =
        cms::alpakatools::make_device_view(queue, otRecHitsSoA.view().detectorIndex(), nTrackerHits);
    alpaka::memcpy(queue, detectorIndexOutputPixel, detectorIndexPixel);
    alpaka::memcpy(queue, detectorIndexOutputTracker, detectorIndexTracker);

    auto offsetBPIX2Output = cms::alpakatools::make_device_view(queue, outputSoA.view().offsetBPIX2());
    auto offsetBPIX2Pixel = cms::alpakatools::make_device_view(queue, pixelRecHitsSoA.view().offsetBPIX2());
    alpaka::memcpy(queue, offsetBPIX2Output, offsetBPIX2Pixel);

    // copy the moduleStart from pixelRecHitsSoA and otRecHitsSoA to outputSoA
    const int nPixelModules = pixelRecHitsSoA.nModules();
    const int nTrackerModules = otRecHitsSoA.nModules() + 1;
    // size of the copy nPixelModules + nTrackerModules + 1 to account for the last "hidden"
    // element of the SoA, keeping track of the cumulative sum of all the hits in the previous
    // modules (thus one more than the number of modules is required to account for the hits
    // in the last tracker module) see also DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h

    auto hitModuleStartOutputPixel =
        cms::alpakatools::make_device_view(queue, outputSoA.view<::reco::HitModuleSoA>().moduleStart(), nPixelModules);
    auto hitModuleStartOutputTracker = cms::alpakatools::make_device_view(
        queue, outputSoA.view<::reco::HitModuleSoA>().moduleStart() + nPixelModules, nTrackerModules);

    const auto hitModuleStartPixel = cms::alpakatools::make_device_view(
        queue, pixelRecHitsSoA.view<::reco::HitModuleSoA>().moduleStart(), nPixelModules);
    const auto hitModuleStartTracker = cms::alpakatools::make_device_view(
        queue, otRecHitsSoA.view<::reco::HitModuleSoA>().moduleStart(), nTrackerModules);

    alpaka::memcpy(queue, hitModuleStartOutputPixel, hitModuleStartPixel);
    alpaka::memcpy(queue, hitModuleStartOutputTracker, hitModuleStartTracker);

    outputSoA.updateFromDevice(queue);
    // emplace the merged SoA in the event
    iEvent.emplace(outputRecHitsSoAToken_, std::move(outputSoA));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiPixelRecHitExtendedAlpaka);