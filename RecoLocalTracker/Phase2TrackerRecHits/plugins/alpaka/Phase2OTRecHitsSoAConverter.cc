#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include <cstdint>
#include <vector>

//#define HITS_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class Phase2OTRecHitsSoAConverter : public stream::EDProducer<> {
    using Hits = ::reco::TrackingRecHitHost;
    using HMSstorage = typename std::vector<uint32_t>;

  public:
    explicit Phase2OTRecHitsSoAConverter(const edm::ParameterSet& iConfig);
    ~Phase2OTRecHitsSoAConverter() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void beginRun(edm::Run const& run, edm::EventSetup const& setup) override;

  private:
    void produce(device::Event& iEvent, const device::EventSetup& es) override;

    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomTokenRun_;
    const edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> recHitToken_;
    const edm::EDGetTokenT<::reco::BeamSpot> beamSpotToken_;
    const edm::EDGetTokenT<Hits> pixelHitsSoA_;

    const edm::EDPutTokenT<Hits> stripSoA_;
    const edm::EDPutTokenT<HMSstorage> hitModuleStart_;

    int modulesInPixel_;
    std::unordered_map<uint32_t, bool> detIdIsP_;
    std::vector<int> orderedModules_;
    std::unordered_map<int, int> moduleIndexToOffset_;
    std::map<uint32_t, uint16_t> detIdToIndex_;
  };

  Phase2OTRecHitsSoAConverter::Phase2OTRecHitsSoAConverter(const edm::ParameterSet& iConfig)
      : stream::EDProducer<>(iConfig),
        geomToken_(esConsumes()),
        geomTokenRun_(esConsumes<edm::Transition::BeginRun>()),
        recHitToken_{consumes(iConfig.getParameter<edm::InputTag>("otRecHitSource"))},
        beamSpotToken_(consumes<::reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
        pixelHitsSoA_{consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSoASource"))},
        stripSoA_{produces()},
        hitModuleStart_{produces()},
        modulesInPixel_(0) {}

  void Phase2OTRecHitsSoAConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("pixelRecHitSoASource", edm::InputTag("hltPhase2SiPixelRecHitsSoA"));
    desc.add<edm::InputTag>("otRecHitSource", edm::InputTag("hltSiPhase2RecHits"));
    desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));

    descriptions.addWithDefaultLabel(desc);
  }

  void Phase2OTRecHitsSoAConverter::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
    detIdIsP_.clear();
    orderedModules_.clear();
    moduleIndexToOffset_.clear();
    detIdToIndex_.clear();
    modulesInPixel_ = 0;

    const auto& trackerGeometry = &iSetup.getData(geomTokenRun_);
    auto isPinPSinOTBarrel = [&](DetId detId) {
      LogDebug("Phase2OTRecHitsSoAConverter")
          << (int)trackerGeometry->getDetectorType(detId) << " "
          << (trackerGeometry->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP) << '\n'
          << (int)detId.subdetId() << " " << (detId.subdetId() == StripSubdetector::TOB) << '\n';
      // Select only P-hits from the OT barrel
      return (trackerGeometry->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP &&
              detId.subdetId() == StripSubdetector::TOB);
    };
    auto isPh2Pixel = [&](DetId detId) {
      auto subId = detId.subdetId();
      return (subId == PixelSubdetector::PixelBarrel || subId == PixelSubdetector::PixelEndcap);
    };

    const auto& detUnits = trackerGeometry->detUnits();

    for (auto& detUnit : detUnits) {
      DetId detId(detUnit->geographicalId());
      detIdIsP_[detId.rawId()] = isPinPSinOTBarrel(detId);
      if (isPh2Pixel(detId))
        modulesInPixel_++;
      if (detIdIsP_[detId.rawId()]) {
        detIdToIndex_[detUnit->geographicalId()] = detUnit->index();
        moduleIndexToOffset_[detUnit->index()] = orderedModules_.size();
        orderedModules_.push_back(detUnit->index());
        LogDebug("Phase2OTRecHitsSoAConverter") << "Inserted " << detUnit->index() << " " << orderedModules_.size()
                                                << " on layer " << int((detId.rawId() >> 20) & 0xF) << '\n';
      }
    }
  }

  void Phase2OTRecHitsSoAConverter::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    auto queue = iEvent.queue();
    auto& bs = iEvent.get(beamSpotToken_);
    const auto& trackerGeometry = &iSetup.getData(geomToken_);
    const auto& stripHits = iEvent.get(recHitToken_);

    const auto& pixelHitsSoA = iEvent.get(pixelHitsSoA_);
    int nPixelHits = pixelHitsSoA.view().metadata().size();

    // Count strip hits and active strip modules
    const int nStripHits = stripHits.data().size();
    const int activeStripModules = stripHits.size();

    // Count the number of P hits in the OT to dimension the SoA
    int PHitsInOTBarrel = 0;
    for (const auto& detSet : stripHits) {
      for (const auto& recHit : detSet) {
        DetId detId(recHit.geographicalId());
        if (detIdIsP_[detId.rawId()])
          PHitsInOTBarrel++;
      }
    }
    LogDebug("Phase2OTRecHitsSoAConverter")
        << "Tot number of modules in Pixels " << modulesInPixel_ << '\n'
        << "Tot number of p_modulesInPSInOTBarrel: " << orderedModules_.size() << '\n'
        << "Number of strip (active) modules:      " << activeStripModules << '\n'
        << "Number of strip hits: " << nStripHits << '\n'
        << "Total hits of PinOTBarrel:   " << PHitsInOTBarrel << '\n';

    Hits stripHitsSoA(queue, PHitsInOTBarrel, orderedModules_.size());
    auto& stripHitsModuleView = stripHitsSoA.view<::reco::HitModuleSoA>();

    std::vector<int> counterOfHitsPerModule(orderedModules_.size(), 0);
    assert(!orderedModules_.empty());
    for (const auto& detSet : stripHits) {
      auto firstHit = detSet.begin();
      auto detId = firstHit->rawId();
      auto index = detIdToIndex_[detId];
      int offset = 0;
      if (detIdIsP_[detId]) {
        offset = moduleIndexToOffset_[index];
        counterOfHitsPerModule[offset] = detSet.size();
      }
    }
#ifdef HITS_DEBUG
    int modId = 0;
    for (auto c : counterOfHitsPerModule) {
      std::cout << "On module " << modId << " we have " << c << " hits." << std::endl;
      modId++;
    }
#endif

    std::vector<int> cumulativeHitPerModule(counterOfHitsPerModule.size());
    std::partial_sum(counterOfHitsPerModule.begin(), counterOfHitsPerModule.end(), cumulativeHitPerModule.begin());
    stripHitsModuleView[0].moduleStart() = nPixelHits;
    LogDebug("Phase2OTRecHitsSoAConverter")
        << "Module start: 0 with hits: " << stripHitsModuleView[0].moduleStart() << '\n';
    for (size_t i = 1; i < cumulativeHitPerModule.size(); ++i) {
      stripHitsModuleView[i].moduleStart() = cumulativeHitPerModule[i - 1] + nPixelHits;
      LogDebug("Phase2OTRecHitsSoAConverter")
          << "Module start: " << i << " with hits: " << stripHitsModuleView[i].moduleStart() << '\n';
    }

    for (const auto& detSet : stripHits) {
      auto firstHit = detSet.begin();
      auto detId = firstHit->rawId();
      auto det = trackerGeometry->idToDet(detId);
      auto index = detIdToIndex_[detId];
      int offset = 0;
      if (detIdIsP_[detId]) {
        offset = moduleIndexToOffset_[index];
        // Start with 0 with no offset, then shifted contents from counterOfHitsPerModule
        int moduleHitIndex = (offset == 0 ? 0 : cumulativeHitPerModule[offset - 1]);
        for (const auto& recHit : detSet) {
          // Select only P-hits from the OT barrel
          if (detIdIsP_[detId]) {
            int idx = moduleHitIndex++;
            assert(idx < PHitsInOTBarrel);
            auto hit = stripHitsSoA.view()[idx];
            hit.xLocal() = recHit.localPosition().x();
            hit.yLocal() = recHit.localPosition().y();
            hit.xerrLocal() = recHit.localPositionError().xx();
            hit.yerrLocal() = recHit.localPositionError().yy();
            auto globalPosition = det->toGlobal(recHit.localPosition());
            double gx = globalPosition.x() - bs.x0();
            double gy = globalPosition.y() - bs.y0();
            double gz = globalPosition.z() - bs.z0();
            hit.xGlobal() = gx;
            hit.yGlobal() = gy;
            hit.zGlobal() = gz;
            hit.rGlobal() = sqrt(gx * gx + gy * gy);
            hit.iphi() = unsafe_atan2s<7>(gy, gx);
            hit.chargeAndStatus().charge = 0;
            hit.chargeAndStatus().status = {false, false, false, false, 0};
            hit.clusterSizeX() = -1;
            hit.clusterSizeY() = -1;
            hit.detectorIndex() = modulesInPixel_ + offset;
            LogDebug("Phase2OTRecHitsSoAConverter")
                << "Local (x, y) with (xx, yy) --> (" << recHit.localPosition().x() << ", "
                << recHit.localPosition().y() << ") with (" << recHit.localPositionError().xx() << ", "
                << recHit.localPositionError().yy() << ")" << '\n'
                << "Global           (x, y, z) --> (" << globalPosition.x() << ", " << globalPosition.y() << ", "
                << globalPosition.z() << ")" << '\n'
                << "Corrected Global (x, y, z) --> (" << gx << ", " << gy << ", " << gz << ")" << '\n'
                << gx << '\n';
          }
        }
      }
    }
    stripHitsModuleView[orderedModules_.size()].moduleStart() =
        cumulativeHitPerModule[orderedModules_.size() - 1] + nPixelHits;

#ifdef HITS_DEBUG
    int current = 0;
    for (int h = 0; h < stripHitsSoA.view().metadata().size(); ++h) {
      auto idx = stripHitsSoA.view()[h].detectorIndex();
      std::cout << h << " detectorIndexInSoA: " << idx << std::endl;
      assert(idx >= current);
      current = idx;
    }
    for (int h = 0; h < stripHitsModuleView.metadata().size(); ++h) {
      std::cout << h << " -> " << stripHitsModuleView[h].moduleStart() << std::endl;
    }
#endif

    HMSstorage moduleStartVec(stripHitsModuleView.metadata().size());

    // Put in the event the hit module start vector.
    // Now, this could  be avoided having the Host Hit SoA
    // consumed by the downstream module (converters to legacy formats).
    // But this is the common practice at the moment
    // also for legacy data formats.
    std::memcpy(moduleStartVec.data(),
                stripHitsModuleView.moduleStart().data(),
                sizeof(uint32_t) * stripHitsModuleView.metadata().size());
    iEvent.emplace(hitModuleStart_, std::move(moduleStartVec));

    // Put the strip hits SoA (host) in the event
    // The framework will take care of the host to device transfer
    iEvent.emplace(stripSoA_, std::move(stripHitsSoA));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(Phase2OTRecHitsSoAConverter);
