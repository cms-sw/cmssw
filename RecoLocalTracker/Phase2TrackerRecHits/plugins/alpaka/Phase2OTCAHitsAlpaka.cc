#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/alpaka/Phase2OTCAModuleParamsCollection.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"

#include "Phase2OTCAHitsKernels.h"
#include "Phase2OTCAModules.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {
    struct CAModules {
      std::unordered_map<uint32_t, uint16_t> moduleByDetId;  // CA module of each DetId of a CA module
      uint16_t nPixelModules = 0;                            // the CA module i has detectorIndex nPixelModules + i
    };
  }  // namespace

  // The OT hits of the CA (CAHitNtupletAlpakaPhase2OT) made on the device from the clusters of the CA modules, and on
  // the host the module starts and a reference to those clusters, for the pixel-track converter to map its OT rechits
  class Phase2OTCAHitsAlpaka : public global::EDProducer<edm::RunCache<CAModules>> {
  public:
    explicit Phase2OTCAHitsAlpaka(edm::ParameterSet const& iConfig)
        : EDProducer<edm::RunCache<CAModules>>(iConfig),
          clustersToken_{consumes(iConfig.getParameter<edm::InputTag>("otClusters"))},
          pixelHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSoASource"))},
          beamSpotToken_{consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))},
          moduleParamsToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("moduleParams"))},
          geometryToken_{esConsumes<edm::Transition::BeginRun>()},
          hitsToken_{produces()},
          moduleStartToken_{produces()},
          clustersRefProdToken_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("otClusters", edm::InputTag("siPhase2Clusters"));
      desc.add<edm::InputTag>("pixelRecHitSoASource", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"))
          ->setComment("pixel rechits on the device: the OT hits follow them in the CA hit indices");
      desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
      desc.add<edm::ESInputTag>("moduleParams", edm::ESInputTag())
          ->setComment("parameters of the CA modules on the device (Phase2OTCAModuleParamsESProducerAlpaka)");
      descriptions.addWithDefaultLabel(desc);
    }

    std::shared_ptr<CAModules> globalBeginRun(edm::Run const&, edm::EventSetup const& iSetup) const override {
      auto const& geometry = iSetup.getData(geometryToken_);
      uint32_t nPixelModules = 0;
      for (auto const* detUnit : geometry.detUnits()) {
        const auto subdetector = detUnit->geographicalId().subdetId();
        if (subdetector == PixelSubdetector::PixelBarrel || subdetector == PixelSubdetector::PixelEndcap)
          ++nPixelModules;
      }
      auto const modules = phase2OTCAHits::caModules(geometry);
      if (modules.empty())
        throw cms::Exception("BadGeometry")
            << "Phase2OTCAHitsAlpaka: the tracker geometry has no OT modules for the CA";
      // the CA hits store nPixelModules + CA module in a uint16_t detectorIndex
      if (nPixelModules + modules.size() > std::numeric_limits<uint16_t>::max() + 1ul)
        throw cms::Exception("BadGeometry") << "Phase2OTCAHitsAlpaka: " << nPixelModules << " pixel modules and "
                                            << modules.size() << " CA modules do not fit a uint16_t detectorIndex";
      auto caModules = std::make_shared<CAModules>();
      caModules->nPixelModules = static_cast<uint16_t>(nPixelModules);
      for (uint32_t module = 0; module < modules.size(); ++module)
        caModules->moduleByDetId.emplace(modules[module]->geographicalId().rawId(), static_cast<uint16_t>(module));
      return caModules;
    }

    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {}

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      using phase2OTCAHits::Cluster;

      CAModules const& caModules = *runCache(static_cast<edm::Event const&>(iEvent).getRun().index());
      const uint32_t nModules = caModules.moduleByDetId.size();
      auto const& moduleParams = iSetup.getData(moduleParamsToken_);
      // the kernel indexes the module parameters by CA module
      if (static_cast<uint32_t>(moduleParams.const_view().metadata().size()) != nModules)
        throw cms::Exception("Configuration")
            << "Phase2OTCAHitsAlpaka: moduleParams has " << moduleParams.const_view().metadata().size()
            << " modules but the tracker geometry has " << nModules << " CA modules";
      const uint32_t nPixelHits = iEvent.get(pixelHitsToken_).nHits();
      auto const clustersHandle = iEvent.getHandle(clustersToken_);
      auto const& clusters = *clustersHandle;
      auto const& beamSpot = iEvent.get(beamSpotToken_);
      auto& queue = iEvent.queue();

      // the clusters of each CA module
      std::vector<std::span<Phase2TrackerCluster1D const>> moduleClusters(nModules);
      uint32_t nHits = 0;
      for (auto const& detSet : clusters) {
        auto const module = caModules.moduleByDetId.find(detSet.detId());
        if (module == caModules.moduleByDetId.end())
          continue;
        moduleClusters[module->second] = {detSet.begin(), detSet.size()};
        nHits += detSet.size();
      }

      // one CA hit per cluster, in CA module order, after the pixel hits in the CA hit indices
      auto moduleStart = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nModules + 1);
      uint32_t firstHit = nPixelHits;
      for (uint32_t module = 0; module < nModules; ++module) {
        moduleStart[module] = firstHit;
        firstHit += moduleClusters[module].size();
      }
      moduleStart[nModules] = firstHit;

      reco::TrackingRecHitsSoACollection hits(queue, nHits, nModules);
      alpaka::memcpy(queue,
                     cms::alpakatools::make_device_view(queue, hits.view().hitModules().moduleStart(), nModules + 1),
                     moduleStart);
      // not used for the OT hits
      auto offsetBPIX2 = cms::alpakatools::make_device_view(queue, hits.view().trackingHits().offsetBPIX2());
      alpaka::memset(queue, offsetBPIX2, 0);
      if (nHits > 0) {
        auto clustersHost = cms::alpakatools::make_host_buffer<Cluster[]>(queue, nHits);
        uint32_t hit = 0;
        for (uint32_t module = 0; module < nModules; ++module)
          for (auto const& cluster : moduleClusters[module])
            clustersHost[hit++] = {static_cast<uint16_t>(cluster.firstStrip()),
                                   static_cast<uint16_t>(cluster.column()),
                                   cluster.size(),
                                   static_cast<uint16_t>(module)};
        auto clustersDevice = cms::alpakatools::make_device_buffer<Cluster[]>(queue, nHits);
        alpaka::memcpy(queue, clustersDevice, clustersHost);
        phase2OTCAHits::makeCAHits(queue,
                                   moduleParams.const_view(),
                                   clustersDevice.data(),
                                   {beamSpot.x0(), beamSpot.y0(), beamSpot.z0()},
                                   caModules.nPixelModules,
                                   hits.view().trackingHits());
      }
      iEvent.emplace(hitsToken_, std::move(hits));
      iEvent.emplace(moduleStartToken_, moduleStart.data(), moduleStart.data() + nModules + 1);
      iEvent.emplace(clustersRefProdToken_, clustersHandle);
    }

  private:
    const edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> clustersToken_;
    const device::EDGetToken<reco::TrackingRecHitsSoACollection> pixelHitsToken_;
    const edm::EDGetTokenT<::reco::BeamSpot> beamSpotToken_;
    const device::ESGetToken<Phase2OTCAModuleParamsCollection, TkPhase2OTCPERecord> moduleParamsToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
    const device::EDPutToken<reco::TrackingRecHitsSoACollection> hitsToken_;
    const edm::EDPutTokenT<std::vector<uint32_t>> moduleStartToken_;
    const edm::EDPutTokenT<edm::RefProd<Phase2TrackerCluster1DCollectionNew>> clustersRefProdToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(Phase2OTCAHitsAlpaka);
