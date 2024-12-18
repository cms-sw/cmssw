
// #include <cstdint>
// #include <memory>
// #include <vector>
// #include "DataFormats/BeamSpot/interface/BeamSpot.h"
// #include "DataFormats/GeometryVector/interface/GlobalPoint.h"
// #include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
// #include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// #include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
// #include "FWCore/ParameterSet/interface/ParameterSet.h"
// #include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
// #include "FWCore/Utilities/interface/InputTag.h"

// // #include "DataFormats/BeamSpot/interface/BeamSpot.h"
// #include "DataFormats/Common/interface/DetSetVectorNew.h"
// #include "DataFormats/Common/interface/Handle.h"
// // #include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
// // #include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
// #include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
// #include "DataFormats/Math/interface/approx_atan2.h"

// #include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
// #include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
// // #include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
// // #include "Geometry/CommonTopologies/interface/GluedGeomDet.h"

// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

// #include "DataFormats/TrackerCommon/interface/TrackerTopology.h"


// namespace ALPAKA_ACCELERATOR_NAMESPACE {

// class TrackingRecHitSoAMerger : public stream::EDProducer<> {

//   using Hits = reco::TrackingRecHitsSoACollection;
//   using HitsHost = ::reco::TrackingRecHitHost;
// public:
//   explicit TrackingRecHitSoAMerger(const edm::ParameterSet& iConfig);
//   ~TrackingRecHitSoAMerger() override = default;

//   static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

// private:
//   void produce(device::Event& iEvent, const device::EventSetup& es) override;
  
//   const device::EDGetTokenT<Hits> hitSoAInOne_;
//   const device::EDGetTokenT<Hits> hitSoAInTwo_;

//   const device::EDPutToken<Hits> hitSoAOut_;
// };


// TrackingRecHitSoAMerger::TrackingRecHitSoAMerger(const edm::ParameterSet& iConfig)
//     : hitSoAInOne_(consumes(iConfig.getParameter<edm::InputTag>("hitSoAOne"))),
//       hitSoAInTwo_(consumes(iConfig.getParameter<edm::InputTag>("hitSoATwo"))),
//       hitSoAOut_{produces()}
// {
  
// }


// void TrackingRecHitSoAMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//   edm::ParameterSetDescription desc;

//   desc.add<edm::InputTag>("stripRecHitSource", edm::InputTag("siStripMatchedRecHits", "matchedRecHit"));
//   desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
//   descriptions.addWithDefaultLabel(desc);

// }

// //https://github.com/cms-sw/cmssw/blob/3f06ef32d66bd2a7fa04e411fa4db4845193bd3c/RecoTracker/MkFit/plugins/convertHits.h

// void TrackingRecHitSoAMerger::produce(device::Event& iEvent, device::EventSetup const& iSetup) {

//   auto queue = iEvent.queue();

//   auto const& hitsOne = iEvent.get(hitSoAInOne_);
//   auto const& hitsTwo = iEvent.get(hitSoAInTwo_);

//   int totHits = hitsOne.view().metadata().size() + hitsTwo.view().metadata().size();
//   int totModsPlusOne = hitsOne.view<::reco::HitModuleSoA>().metadata().size() + hitsTwo.view<::reco::HitModuleSoA>().metadata().size() - 1;

//   Hits outHits(queue, totHits, totModules); 
//   alpaka::memcpy(queue, outHits.buffer(), hitsOne.buffer());
//   stripHitsDevice.updateFromDevice(queue);


//   iEvent.emplace(hitSoAOut_, std::move(outHits));
//   iEvent.emplace(stripSoADevice_, std::move(stripHitsDevice));
  
// }

// }

// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
// DEFINE_FWK_ALPAKA_MODULE(TrackingRecHitSoAMerger);
