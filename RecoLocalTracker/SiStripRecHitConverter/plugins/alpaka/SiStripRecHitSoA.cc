
#include <cstdint>
#include <memory>
#include <vector>
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/BeamSpot/interface/alpaka/BeamSpotDevice.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/SimplePixelStripTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
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

// #include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/Math/interface/approx_atan2.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/CommonTopologies/interface/GluedGeomDet.h"

#include "SiStripRecHitSoAKernel.h"
#include "alpaka/mem/view/Traits.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

template <typename TrackerTraits>
class SiStripRecHitSoA : public stream::SynchronizingEDProducer<> {

  using PixelBase = typename TrackerTraits::PixelBase;

  using StripHits = TrackingRecHitsSoACollection<TrackerTraits>;
  using PixelHits = TrackingRecHitsSoACollection<PixelBase>;

  using StripHitsHost = TrackingRecHitHost<TrackerTraits>;
  using PixelHitsHost = TrackingRecHitHost<PixelBase>;

  using Algo = hitkernels::SiStripRecHitSoAKernel<TrackerTraits>;

public:
  explicit SiStripRecHitSoA(const edm::ParameterSet& iConfig);
  ~SiStripRecHitSoA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override {};
  void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> recHitToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const edm::EDGetTokenT<PixelHitsHost> pixelRecHitSoAToken_;

  const device::EDPutToken<StripHits> stripSoA_;
  const edm::EDPutTokenT<std::vector<uint32_t>> hmsToken_;

  const Algo Algo_;
};

template <typename TrackerTraits>
SiStripRecHitSoA<TrackerTraits>::SiStripRecHitSoA(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      recHitToken_{consumes(iConfig.getParameter<edm::InputTag>("stripRecHitSource"))},
      //beamSpotToken(consumes(edm::InputTag("offlineBeamSpot"))),
      beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      pixelRecHitSoAToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSoASource"))},
      stripSoA_{produces()},
      hmsToken_{produces()}
{
  
}

template <typename TrackerTraits>
void SiStripRecHitSoA<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("stripRecHitSource", edm::InputTag("siStripMatchedRecHits", "matchedRecHit"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("pixelRecHitSoASource", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));
  descriptions.addWithDefaultLabel(desc);

  // desc.setUnknown();
  // descriptions.addDefault(desc);
}

template <typename TrackerTraits>
void SiStripRecHitSoA<TrackerTraits>::produce(device::Event& iEvent, device::EventSetup const& iSetup) {

  // Get the objects that we need
  const TrackerGeometry* trackerGeometry = &iSetup.getData(geomToken_);
  auto const& stripHits = iEvent.get(recHitToken_);
  auto const& pixelHitsHost = iEvent.get(pixelRecHitSoAToken_);
  auto& bs = iEvent.get(beamSpotToken_);

  // Count strip hits
  size_t nStripHits = 0;
  //std::cout << "number of modules: " << TrackerTraits::numberOfModules << std::endl;
  //std::cout << "stripHits size: " << stripHits.size() << std::endl;
  for (const auto& detSet : stripHits) {
    const GluedGeomDet* det = static_cast<const GluedGeomDet*>(trackerGeometry->idToDet(detSet.detId()));
    //std::cout << "detSet.detId()" << detSet.detId() << std::endl;
    //std::cout << "det->stereoDet()->index()" << det->stereoDet()->index() << std::endl;
    if (TrackerTraits::mapIndex(det->stereoDet()->index()) < TrackerTraits::numberOfModules)
        nStripHits += detSet.size();
  } 

  size_t nPixelHits = pixelHitsHost.view().metadata().size();

  //std::cout << "nStripHits = " << nStripHits << std::endl;
  //std::cout << "nPixelHits = " << nPixelHits << std::endl;

  // HostView<const PixelHits, PixelHitsHost> pixelHitsHostView(pixelHits, iEvent.queue());
  // PixelHitsHost& pixelHitsHost = pixelHitsHostView.get();
  // PixelHitsHost pixelHitsHost(nPixelHits, iEvent.queue());

  // alpaka::memcpy(iEvent.queue(), pixelHitsHost.buffer(), pixelHits.buffer());

  StripHitsHost allHitsHost(
    iEvent.queue(),
    nPixelHits + nStripHits
  );
  
  // Copy pixel data
  std::copy(pixelHitsHost.view().xLocal(), pixelHitsHost.view().xLocal() + nPixelHits, allHitsHost.view().xLocal());
  std::copy(pixelHitsHost.view().yLocal(), pixelHitsHost.view().yLocal() + nPixelHits, allHitsHost.view().yLocal());
  std::copy(pixelHitsHost.view().xerrLocal(), pixelHitsHost.view().xerrLocal() + nPixelHits, allHitsHost.view().xerrLocal());
  std::copy(pixelHitsHost.view().yerrLocal(), pixelHitsHost.view().yerrLocal() + nPixelHits, allHitsHost.view().yerrLocal());
  std::copy(pixelHitsHost.view().xGlobal(), pixelHitsHost.view().xGlobal() + nPixelHits, allHitsHost.view().xGlobal());
  std::copy(pixelHitsHost.view().yGlobal(), pixelHitsHost.view().yGlobal() + nPixelHits, allHitsHost.view().yGlobal());
  std::copy(pixelHitsHost.view().zGlobal(), pixelHitsHost.view().zGlobal() + nPixelHits, allHitsHost.view().zGlobal());
  std::copy(pixelHitsHost.view().rGlobal(), pixelHitsHost.view().rGlobal() + nPixelHits, allHitsHost.view().rGlobal());
  std::copy(pixelHitsHost.view().iphi(), pixelHitsHost.view().iphi() + nPixelHits, allHitsHost.view().iphi());
  std::copy(pixelHitsHost.view().chargeAndStatus(), pixelHitsHost.view().chargeAndStatus() + nPixelHits, allHitsHost.view().chargeAndStatus());
  std::copy(pixelHitsHost.view().clusterSizeX(), pixelHitsHost.view().clusterSizeX() + nPixelHits, allHitsHost.view().clusterSizeX());
  std::copy(pixelHitsHost.view().clusterSizeY(), pixelHitsHost.view().clusterSizeY() + nPixelHits, allHitsHost.view().clusterSizeY());
  std::copy(pixelHitsHost.view().detectorIndex(), pixelHitsHost.view().detectorIndex() + nPixelHits, allHitsHost.view().detectorIndex());

  std::copy(pixelHitsHost.view().phiBinnerStorage(), pixelHitsHost.view().phiBinnerStorage() + nPixelHits, allHitsHost.view().phiBinnerStorage());

  allHitsHost.view().offsetBPIX2() = pixelHitsHost.view().offsetBPIX2();

  auto& hitsModuleStart = allHitsHost.view().hitsModuleStart();

  std::copy(
    pixelHitsHost.view().hitsModuleStart().begin(),
    pixelHitsHost.view().hitsModuleStart().end(),
    hitsModuleStart.begin()
  );  

  std::map<size_t, edmNew::DetSet<SiStripMatchedRecHit2D>> mappedModuleHits;

  // Loop over strip RecHits
  for (auto detSet : stripHits) {

    const GluedGeomDet* det = static_cast<const GluedGeomDet*>(trackerGeometry->idToDet(detSet.detId()));
    size_t index = TrackerTraits::mapIndex(det->stereoDet()->index());
    
    if (index >= TrackerTraits::numberOfModules)
      continue;

    mappedModuleHits[index] = detSet;
  }

  size_t i = 0;
  size_t lastIndex = TrackerTraits::numberOfPixelModules;

  for (auto& [index, detSet] : mappedModuleHits) {

    const GluedGeomDet* det = static_cast<const GluedGeomDet*>(trackerGeometry->idToDet(detSet.detId()));

    // no hits since lastIndex: hitsModuleStart[lastIndex:index] = hitsModuleStart[lastIndex]
    for (auto j = lastIndex + 1; j < index + 1; ++j)
      hitsModuleStart[j] = hitsModuleStart[lastIndex];

    hitsModuleStart[index + 1] = hitsModuleStart[index] + detSet.size();
    lastIndex = index + 1;

    for (const auto& recHit : detSet) {
      allHitsHost.view()[nPixelHits + i].xLocal() = recHit.localPosition().x();
      allHitsHost.view()[nPixelHits + i].yLocal() = recHit.localPosition().y();
      allHitsHost.view()[nPixelHits + i].xerrLocal() = recHit.localPositionError().xx();
      allHitsHost.view()[nPixelHits + i].yerrLocal() = recHit.localPositionError().yy();
      auto globalPosition = det->toGlobal(recHit.localPosition());
      double gx = globalPosition.x() - bs.x0();
      double gy = globalPosition.y() - bs.y0();
      double gz = globalPosition.z() - bs.z0();
      allHitsHost.view()[nPixelHits + i].xGlobal() = gx;
      allHitsHost.view()[nPixelHits + i].yGlobal() = gy;
      allHitsHost.view()[nPixelHits + i].zGlobal() = gz;
      allHitsHost.view()[nPixelHits + i].rGlobal() = sqrt(gx * gx + gy * gy);
      allHitsHost.view()[nPixelHits + i].iphi() = unsafe_atan2s<7>(gy, gx);
      // allHitsHost.view()[nPixelHits + i].chargeAndStatus().charge = ?
      // allHitsHost.view()[nPixelHits + i].chargeAndStatus().status = ?
      // allHitsHost.view()[nPixelHits + i].clusterSizeX() = ?
      // allHitsHost.view()[nPixelHits + i].clusterSizeY() = ?
      allHitsHost.view()[nPixelHits + i].detectorIndex() = index;
      // ???
      ++i;
    }
    
  }

  for (auto j = lastIndex + 1; j < TrackerTraits::numberOfModules + 1; ++j)
    hitsModuleStart[j] = hitsModuleStart[lastIndex];


  for (auto layer = 0U; layer < TrackerTraits::numberOfLayers + 1; ++layer) {
    allHitsHost.view().hitsLayerStart()[layer] = 
      hitsModuleStart[TrackerTraits::layerStart[layer]];
  }

  iEvent.emplace(hmsToken_, std::vector<uint32_t>(hitsModuleStart.begin(), hitsModuleStart.end()));

  iEvent.emplace(stripSoA_, Algo_.fillHitsAsync(allHitsHost, iEvent.queue()));

  //std::cout << "produce done" << std::endl;
  
}
  using SiStripRecHitSoAPhase1 = SiStripRecHitSoA<pixelTopology::Phase1Strip>;
}

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiStripRecHitSoAPhase1);

// using SiPixelRecHitSoAFromLegacyPhase2 = SiStripRecHitSoA<pixelTopology::Phase2>;
// DEFINE_FWK_MODULE(SiPixelRecHitSoAFromLegacyPhase2);
