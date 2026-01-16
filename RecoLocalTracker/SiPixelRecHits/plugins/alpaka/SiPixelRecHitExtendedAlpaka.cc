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
#include "RecoLocalTracker/Records/interface/PixelCPEFastParamsRecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/alpaka/PixelCPEFastParamsCollection.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

//#define GPU_DEBUG

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

    desc.add<edm::InputTag>("pixelRecHitsSoA", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));
    desc.add<edm::InputTag>("trackerRecHitsSoA", edm::InputTag("phase2OTRecHitsSoAConverter"));

    descriptions.addWithDefaultLabel(desc);
  }

  namespace {
    // This utility unrolls the SoA columns (tuples) at compile time, calling the provided functor 'f'
    // once for each element. The index is passed as a std::integral_constant so it
    // is available at compile time.
    template <typename F, std::size_t... Is>
    void unrollColumns(F&& f, std::index_sequence<Is...>) {
      (f(std::integral_constant<std::size_t, Is>{}), ...);
    }
    // User-facing wrapper to deduce the size of the tuple and create the index sequence
    // Usage: mergeSoAColumns<NumberOfColumns>([&](auto columnIndex) { ... });
    template <std::size_t N, typename F>
    void mergeSoAColumns(F&& f) {
      unrollColumns(std::forward<F>(f), std::make_index_sequence<N>{});
    }
  }  // namespace

  void SiPixelRecHitExtendedAlpaka::produce(edm::StreamID streamID,
                                            device::Event& iEvent,
                                            const device::EventSetup& es) const {
    // get both Pixel and Tracker SoA collections
    auto queue = iEvent.queue();
    const auto& pixColl = iEvent.get(pixelRecHitToken_);
    const auto& trkColl = iEvent.get(trackerRecHitToken_);

    // pix and trk SoA collections have the same layout
    // each of them is made up of two SoAs:
    // - one that contains the hits
    // - one to track the number of hits in each module (hitModuleSoA)
    // this code merges and copy both of them into a new SoA collection
    // taking into account that for the hits the copy is straightforward,
    // while for the hitModuleSoA we need to copy nPixelModules + nTrackerModules + 1 elements
    // to account for the last "hidden" element in the SoA which is used to store
    // the cumulative sum of hits in all the previous modules (thus this SoA has 1 more
    // element than the actual number of modules to track the hits in the last module
    // and sum them to the others).
    // See also DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h

    const int nPixHits = pixColl.nHits();
    const int nTrkHits = trkColl.nHits();

    const int nPixMod = pixColl.nModules();
    const int nTrkMod = trkColl.nModules();

    // the output is also a SoA collection with the same layout as the input ones
    auto output = reco::TrackingRecHitsSoACollection(queue, nPixHits + nTrkHits, nPixMod + nTrkMod);

#ifdef GPU_DEBUG
    std::cout << "----------------- Merging Pixel and Tracker RecHits -----------------\n"
              << "Number of Pixel recHits: " << nPixHits << '\n'
              << "Number of Tracker recHits: " << nTrkHits << '\n'
              << "Total number of recHits: " << output.nHits() << '\n'
              << "Number of Pixel modules: " << nPixMod << '\n'
              << "Number of Tracker modules: " << nTrkMod << '\n'
              << "Total number of modules: " << output.nModules() << '\n'
              << "---------------------------------------------------------------------\n";
#endif

    // start from the hits SoA, use metarecords to loop over all the columns
    auto outView = output.view();
    auto pixView = pixColl.view();
    auto trkView = trkColl.view();

    // layout type (same for all views)
    using ViewType = decltype(outView);
    using LayoutType = typename ViewType::Metadata::TypeOf_Layout;

    // build descriptors (tuple of spans: one span for each column)
    auto outDesc = LayoutType::Descriptor(outView);
    auto pixDesc = LayoutType::ConstDescriptor(pixView);
    auto trkDesc = LayoutType::ConstDescriptor(trkView);

    // merge all columns using a compile-time loop

    // number of columns (same for all hits SoAs)
    constexpr std::size_t N = std::tuple_size_v<decltype(outDesc.buff)>;
    mergeSoAColumns<N>([&](auto columnIndex) {
      auto& outCol = std::get<columnIndex>(outDesc.buff);
      const auto& pixCol = std::get<columnIndex>(pixDesc.buff);
      const auto& trkCol = std::get<columnIndex>(trkDesc.buff);
      // distinguish between scalar and column types
      if constexpr (std::get<columnIndex>(outDesc.columnTypes) == cms::soa::SoAColumnType::scalar) {
        // scalar type, copy the value directly
        alpaka::memcpy(queue,
                       cms::alpakatools::make_device_view(queue, outCol.data(), 1),
                       cms::alpakatools::make_device_view(queue, pixCol.data(), 1));
#ifdef GPU_DEBUG
        alpaka::wait(queue);
        std::cout << "Copied scalar with index " << columnIndex << '\n';
#endif
      } else {
        // column type, copy the whole column
        // copy Pixel hits
        alpaka::memcpy(queue,
                       cms::alpakatools::make_device_view(queue, outCol.data(), nPixHits),
                       cms::alpakatools::make_device_view(queue, pixCol.data(), nPixHits));
        // copy Tracker hits (offset after Pixel hits)
        alpaka::memcpy(queue,
                       cms::alpakatools::make_device_view(queue, outCol.data() + nPixHits, nTrkHits),
                       cms::alpakatools::make_device_view(queue, trkCol.data(), nTrkHits));
#ifdef GPU_DEBUG
        alpaka::wait(queue);
        std::cout << "Copied column with index " << columnIndex << '\n';
#endif
      }
    });
    // copy hitModuleStart for Pixel modules
    alpaka::memcpy(
        queue,
        cms::alpakatools::make_device_view(queue, output.view<::reco::HitModuleSoA>().moduleStart().data(), nPixMod),
        cms::alpakatools::make_device_view(queue, pixColl.view<::reco::HitModuleSoA>().moduleStart().data(), nPixMod));
    // copy hitModuleStart for Tracker modules (offset after Pixel modules)
    // copy nTrkMod + 1 elements to include the last "hidden" element
    alpaka::memcpy(queue,
                   cms::alpakatools::make_device_view(
                       queue, output.view<::reco::HitModuleSoA>().moduleStart().data() + nPixMod, nTrkMod + 1),
                   cms::alpakatools::make_device_view(
                       queue, trkColl.view<::reco::HitModuleSoA>().moduleStart().data(), nTrkMod + 1));
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Copied hitModuleStart for Pixel and Tracker modules\n";
#endif

    // update the information cached in the output collection with device information
    output.updateFromDevice(queue);

    // emplace the merged SoA collection in the event
    iEvent.emplace(outputRecHitsSoAToken_, std::move(output));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiPixelRecHitExtendedAlpaka);
