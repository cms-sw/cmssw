#include <algorithm>
#include <iterator>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/StubsSoACollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  /**
   * @brief Merges pixel RecHits with OT stubs into a unified TrackingRecHitsSoACollection.
   *
   * Stubs are placed at hitIdx = nPixHits + stubIdx so stubIdx = hitIdx - offsetStubs
   * maps back to the StubsSoA. moduleStart is dense: the first nPixelModules entries are the pixel
   * modules, followed by the nOTModulesActual outer-tracker modules the stub collection reports at
   * run time (the compile-time nModulesTotStubs is only used by the GPU_DEBUG assertions).
   */
  class SiPixelRecHitsStubsMerger : public global::EDProducer<> {
  public:
    explicit SiPixelRecHitsStubsMerger(const edm::ParameterSet& iConfig);
    ~SiPixelRecHitsStubsMerger() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    const device::EDGetToken<reco::TrackingRecHitsSoACollection> pixelRecHitToken_;
    const device::EDGetToken<reco::StubsSoACollection> stubsToken_;
    const device::EDGetToken<reco::OTRecHitsSoACollection> otRecHitToken_;

    const device::EDPutToken<reco::TrackingRecHitsSoACollection> outputRecHitsSoAToken_;
  };

  SiPixelRecHitsStubsMerger::SiPixelRecHitsStubsMerger(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        pixelRecHitToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitsSoA"))),
        stubsToken_(consumes(iConfig.getParameter<edm::InputTag>("stubsSoA"))),
        otRecHitToken_(consumes(iConfig.getParameter<edm::InputTag>("otRecHitsSoA"))),
        outputRecHitsSoAToken_(produces()) {}

  void SiPixelRecHitsStubsMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("pixelRecHitsSoA", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));
    desc.add<edm::InputTag>("stubsSoA", edm::InputTag("otStubProducer"));
    desc.add<edm::InputTag>("otRecHitsSoA", edm::InputTag("otRecHitsSoAConverter"));

    descriptions.addWithDefaultLabel(desc);
  }

  namespace {
#ifdef GPU_DEBUG
    constexpr uint32_t nModulesCA = phase2PixelTopology::nModulesTotStubs;  // 17200
#endif
    constexpr uint32_t nPixelModules = phase2PixelTopology::nModulesPix;  // 4000

    // Kernel to initialize stub fields for pixel hits (set to zero/false)
    struct InitializePixelStubFieldsKernel {
      template <typename TAcc, typename HitsView>
      ALPAKA_FN_ACC void operator()(const TAcc& acc, HitsView hitsView, uint32_t nPixHits) const {
        for (uint32_t hitIdx : cms::alpakatools::uniform_elements(acc, nPixHits)) {
          hitsView.dPhiDr()[hitIdx] = 0.0f;
          hitsView.dPhiDrError()[hitIdx] = -1.0f;
          // Set lowerHitIdx to max value (invalid) for pixel hits
          hitsView.lowerHitIdx()[hitIdx] = UINT32_MAX;
          hitsView.stubFlags()[hitIdx] = 0;
        }
      }
    };

    // Kernel to initialize moduleStart array for OT modules
    // Sets all OT entries to point to the end of stub hits (indicating empty)
    struct InitializeOTModuleStartKernel {
      template <typename TAcc, typename ModuleView>
      ALPAKA_FN_ACC void operator()(
          const TAcc& acc, ModuleView moduleView, uint32_t nPixHits, uint32_t nStubs, uint32_t nOTModulesActual) const {
        uint32_t endOffset = nPixHits + nStubs;
        // Initialize OT module entries (4000 to 17200 inclusive)
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, nOTModulesActual + 1)) {
          moduleView.moduleStart()[nPixelModules + i] = endOffset;
        }
      }
    };

    // Kernel to count stubs per CA module
    // Input: stubs in geometry order with detectorIndex set to CA module
    // Output: stubCounts[caModule] = number of stubs in that CA module
    struct CountStubsPerCAModuleKernel {
      template <typename TAcc, typename StubsView, typename OTHitsView>
      ALPAKA_FN_ACC void operator()(
          const TAcc& acc, StubsView stubsView, OTHitsView otHitsView, uint32_t* stubCounts, uint32_t nStubs) const {
        for (uint32_t stubIdx : cms::alpakatools::uniform_elements(acc, nStubs)) {
          auto lowerHitIdx = stubsView.lowerHitIdx()[stubIdx];
          uint32_t caModule = otHitsView.detectorIndex()[lowerHitIdx];
          // Atomic increment for this CA module
          alpaka::atomicAdd(acc, &stubCounts[caModule], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    };

    // Copy stubs to the merged collection preserving StubsSoA order.
    // Stubs at hitIdx = nPixHits + stubIdx; StubsSoA already grouped by module.
    struct CopyStubsToHitsKernel {
      template <typename TAcc, typename StubsView, typename HitsView, typename OTHitsView>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    StubsView stubsView,
                                    HitsView hitsView,
                                    OTHitsView otHitsView,
                                    uint32_t nPixHits,
                                    uint32_t nStubs) const {
        for (uint32_t stubIdx : cms::alpakatools::uniform_elements(acc, nStubs)) {
          uint32_t hitIdx = nPixHits + stubIdx;

          // Copy position information from lower hit
          auto lowerHitIdx = stubsView.lowerHitIdx()[stubIdx];
          hitsView.xLocal()[hitIdx] = otHitsView.xLocal()[lowerHitIdx];
          hitsView.yLocal()[hitIdx] = otHitsView.yLocal()[lowerHitIdx];
          hitsView.xerrLocal()[hitIdx] = otHitsView.xerrLocal()[lowerHitIdx];
          hitsView.yerrLocal()[hitIdx] = otHitsView.yerrLocal()[lowerHitIdx];

          auto xg = otHitsView.xGlobal()[lowerHitIdx];
          auto yg = otHitsView.yGlobal()[lowerHitIdx];
          hitsView.xGlobal()[hitIdx] = xg;
          hitsView.yGlobal()[hitIdx] = yg;
          hitsView.zGlobal()[hitIdx] = otHitsView.zGlobal()[lowerHitIdx];
          hitsView.rGlobal()[hitIdx] = alpaka::math::sqrt(acc, xg * xg + yg * yg);
          hitsView.iphi()[hitIdx] = stubsView.iphi()[stubIdx];

          // Set cluster info (stubs don't have charge)
          hitsView.chargeAndStatus()[hitIdx].charge = 0;
          hitsView.chargeAndStatus()[hitIdx].status.isBigX = false;
          hitsView.chargeAndStatus()[hitIdx].status.isOneX = false;
          hitsView.chargeAndStatus()[hitIdx].status.isBigY = false;
          hitsView.chargeAndStatus()[hitIdx].status.isOneY = false;
          hitsView.chargeAndStatus()[hitIdx].status.qBin = 0;

          hitsView.clusterSizeX()[hitIdx] = 0;
          hitsView.clusterSizeY()[hitIdx] = 0;

          // Set detector index (CA module)
          hitsView.detectorIndex()[hitIdx] = otHitsView.detectorIndex()[lowerHitIdx];

          // Set stub-specific fields
          hitsView.dPhiDr()[hitIdx] = stubsView.dPhiDr()[stubIdx];
          hitsView.dPhiDrError()[hitIdx] = stubsView.dPhiDrError()[stubIdx];
          // Copy lowerHitIdx if PS stub for duplicate stub handling in CAFishbone
          bool isPS = ::reco::StubFlags::isPS(stubsView.flags()[stubIdx]);
          hitsView.lowerHitIdx()[hitIdx] = isPS ? stubsView.lowerHitIdx()[stubIdx] : UINT32_MAX;
          // Copy stub flags (isBarrel, isFlat, isValid, layer) for pairwise compatibility cuts
          hitsView.stubFlags()[hitIdx] = stubsView.flags()[stubIdx];
        }
      }
    };

    // Kernel to build the OT part of moduleStart from the per-module stub counts:
    // exclusive prefix sum of the OT module stub counts, offset by nPixHits.
    struct BuildModuleStartKernel {
      // Launched as ONE block of kChunk threads: chunked block-wide prefix scan.
      static constexpr int32_t kChunk = 1024;

      template <typename TAcc, typename ModuleView>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    ModuleView moduleView,
                                    uint32_t const* stubCounts,
                                    uint32_t nPixHits,
                                    uint32_t nOTModulesActual) const {
        auto& ws = alpaka::declareSharedVar<uint32_t[64], __COUNTER__>(acc);  // warpSize entries (64 covers ROCm)
        auto& ci = alpaka::declareSharedVar<uint32_t[kChunk], __COUNTER__>(acc);
        auto& co = alpaka::declareSharedVar<uint32_t[kChunk], __COUNTER__>(acc);
        auto& carry = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
        if (cms::alpakatools::once_per_block(acc)) {
          carry = nPixHits;
        }
        alpaka::syncBlockThreads(acc);
        for (uint32_t base = 0; base < nOTModulesActual; base += kChunk) {
          const int32_t n = (nOTModulesActual - base < static_cast<uint32_t>(kChunk))
                                ? static_cast<int32_t>(nOTModulesActual - base)
                                : kChunk;
          for (auto i : cms::alpakatools::independent_group_elements(acc, n)) {
            ci[i] = stubCounts[nPixelModules + base + i];
          }
          alpaka::syncBlockThreads(acc);
          cms::alpakatools::blockPrefixScan(acc, ci, co, n, ws);  // inclusive: co[i] = sum_{k <= i} ci[k]
          alpaka::syncBlockThreads(acc);
          const uint32_t c = carry;
          for (auto i : cms::alpakatools::independent_group_elements(acc, n)) {
            moduleView.moduleStart()[nPixelModules + base + i] = c + co[i] - ci[i];
          }
          alpaka::syncBlockThreads(acc);
          if (cms::alpakatools::once_per_block(acc)) {
            carry = c + co[n - 1];
          }
          alpaka::syncBlockThreads(acc);
        }
        if (cms::alpakatools::once_per_block(acc)) {
          moduleView.moduleStart()[nPixelModules + nOTModulesActual] = carry;
        }
      }
    };

  }  // namespace

  void SiPixelRecHitsStubsMerger::produce(edm::StreamID streamID,
                                          device::Event& iEvent,
                                          const device::EventSetup& es) const {
    auto queue = iEvent.queue();
    const auto& pixRecHitsColl = iEvent.get(pixelRecHitToken_);
    const auto& stubsColl = iEvent.get(stubsToken_);
    const auto& otRecHitsColl = iEvent.get(otRecHitToken_);

    const uint32_t nPixHits = pixRecHitsColl.nHits();
    const uint32_t nStubs = stubsColl.nStubs();
    // Actual OT module count from the stubs collection.
    const uint32_t nOTModulesActual = stubsColl.nModules();
    const uint32_t nModulesTotal = nPixelModules + nOTModulesActual;

#ifdef GPU_DEBUG
    std::cout << "----------------- Merging Pixel RecHits and OT Stubs -----------------\n"
              << "Number of Pixel recHits: " << nPixHits << '\n'
              << "Number of OT Stubs: " << nStubs << '\n'
              << "Total number of hits: " << (nPixHits + nStubs) << '\n'
              << "Number of CA modules (hardcoded): " << nModulesCA << '\n'
              << "Number of CA modules (actual): " << nModulesTotal << '\n'
              << "  Pixel modules: " << nPixelModules << '\n'
              << "  OT modules (actual): " << nOTModulesActual << '\n'
              << "offsetStubs will be set to: " << nPixHits << '\n'
              << "----------------------------------------------------------------------\n";
#endif

    auto output = reco::TrackingRecHitsSoACollection(queue, nPixHits + nStubs, nModulesTotal);

    auto outHitsView = output.view().trackingHits();
    auto pixHitsView = pixRecHitsColl.view().trackingHits();
    auto const otHitsView = otRecHitsColl.const_view().otRecHits();

    auto outModuleView = output.view().hitModules();
    auto stubsSubView = stubsColl.view().stubs();

    // Step 1: Copy pixel hits.
    using HitsViewType = decltype(outHitsView);
    using HitsLayoutType = typename HitsViewType::Metadata::TypeOf_Layout;

    auto outDesc = HitsLayoutType::Descriptor(outHitsView);
    auto pixDesc = HitsLayoutType::ConstDescriptor(pixHitsView);

    constexpr std::size_t N = std::tuple_size_v<decltype(outDesc.buff)>;
    // The 19 members of reco::TrackingHitsLayout (17 columns + 2 scalars). A layout change lands here
    // rather than silently skipping the wrong column.
    static_assert(N == 19, "TrackingHitsLayout member count changed: re-check the skip list below");

    // The four STUB-SPECIFIC columns are overwritten in full, over exactly [0, nPixHits), by
    // InitializePixelStubFieldsKernel a few lines below (dPhiDr = 0, dPhiDrError = -1,
    // lowerHitIdx = UINT32_MAX, stubFlags = 0). Copying the pixel collection's values into them
    // first is pure waste: 4 of the 19 D2D copies and megabytes per event of device traffic,
    // every byte of which is overwritten before anything reads it. Both the copy and the
    // init kernel are guarded by nPixHits > 0, so the ranges coincide exactly.
    // The skip is by destination POINTER, not by column index, so it follows the members if the
    // layout is ever reordered; if a pointer ever failed to match, the column would simply be copied
    // as before (no correctness cliff).
    const void* const kInitialisedCols[] = {static_cast<const void*>(outHitsView.dPhiDr().data()),
                                            static_cast<const void*>(outHitsView.dPhiDrError().data()),
                                            static_cast<const void*>(outHitsView.lowerHitIdx().data()),
                                            static_cast<const void*>(outHitsView.stubFlags().data())};

    auto copyColumns = [&](auto columnIndex) {
      auto& outCol = std::get<columnIndex>(outDesc.buff);
      const auto& pixCol = std::get<columnIndex>(pixDesc.buff);

      if constexpr (std::get<columnIndex>(outDesc.columnTypes) == cms::soa::SoAColumnType::scalar) {
        alpaka::memcpy(queue,
                       cms::alpakatools::make_device_view(queue, outCol.data(), 1),
                       cms::alpakatools::make_device_view(queue, pixCol.data(), 1));
      } else {
        const void* const dst = static_cast<const void*>(outCol.data());
        const bool overwrittenByInit = std::any_of(
            std::begin(kInitialisedCols), std::end(kInitialisedCols), [dst](const void* p) { return p == dst; });
        if (nPixHits > 0 && !overwrittenByInit) {
          alpaka::memcpy(queue,
                         cms::alpakatools::make_device_view(queue, outCol.data(), nPixHits),
                         cms::alpakatools::make_device_view(queue, pixCol.data(), nPixHits));
        }
      }
    };

    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      (copyColumns(std::integral_constant<std::size_t, Is>{}), ...);
    }(std::make_index_sequence<N>{});

    // Step 2: Set offsetStubs scalar via memcpy.
    {
      uint32_t offsetStubsValue = nPixHits;
      auto offsetStubsHostView = cms::alpakatools::make_host_view(offsetStubsValue);
      auto offsetStubsDeviceView = cms::alpakatools::make_device_view(queue, outHitsView.offsetStubs());
      alpaka::memcpy(queue, offsetStubsDeviceView, offsetStubsHostView);
    }

    // Step 3: Initialize stub fields for pixel hits.
    constexpr uint32_t threadsPerBlock = 256;
    if (nPixHits > 0) {
      uint32_t blocksInit = (nPixHits + threadsPerBlock - 1) / threadsPerBlock;
      auto workDivInit = cms::alpakatools::make_workdiv<Acc1D>(blocksInit, threadsPerBlock);
      alpaka::exec<Acc1D>(queue, workDivInit, InitializePixelStubFieldsKernel{}, outHitsView, nPixHits);
    }

    // Step 4: Copy pixel moduleStart (entries 0 to nPixelModules).
    // The pixel collection has dense moduleStart indexed by module ID 0-3999
    alpaka::memcpy(queue,
                   cms::alpakatools::make_device_view(queue, outModuleView.moduleStart().data(), nPixelModules + 1),
                   cms::alpakatools::make_device_view(
                       queue, pixRecHitsColl.view().hitModules().moduleStart().data(), nPixelModules + 1));

    // Step 5: Handle OT stubs.
    if (nStubs > 0) {
      auto outModuleView = output.view().hitModules();

      auto stubCounts = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nModulesTotal + 1);

      alpaka::memset(queue, stubCounts, 0);

      {
        uint32_t blocksCount = (nStubs + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocksCount, threadsPerBlock);
        alpaka::exec<Acc1D>(
            queue, workDiv, CountStubsPerCAModuleKernel{}, stubsSubView, otHitsView, stubCounts.data(), nStubs);
      }

      {
        auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, BuildModuleStartKernel::kChunk);
        alpaka::exec<Acc1D>(
            queue, workDiv, BuildModuleStartKernel{}, outModuleView, stubCounts.data(), nPixHits, nOTModulesActual);
      }

      // Copy stubs to the hit collection preserving StubsSoA order (see CopyStubsToHitsKernel).
      {
        uint32_t blocksCopy = (nStubs + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocksCopy, threadsPerBlock);
        alpaka::exec<Acc1D>(
            queue, workDiv, CopyStubsToHitsKernel{}, stubsSubView, outHitsView, otHitsView, nPixHits, nStubs);
      }
    } else {
      // No stubs: initialize OT moduleStart to empty.
      auto outModuleView = output.view().hitModules();
      uint32_t blocksOT = (nOTModulesActual + 1 + threadsPerBlock - 1) / threadsPerBlock;
      auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocksOT, threadsPerBlock);
      alpaka::exec<Acc1D>(
          queue, workDiv, InitializeOTModuleStartKernel{}, outModuleView, nPixHits, 0, nOTModulesActual);
    }

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    // Read offsetStubs from device via memcpy (cannot directly read device memory from host)
    uint32_t offsetStubsDebug = 0;
    auto offsetStubsDebugHostView = cms::alpakatools::make_host_view(offsetStubsDebug);
    auto offsetStubsDebugDeviceView = cms::alpakatools::make_device_view(queue, outHitsView.offsetStubs());
    alpaka::memcpy(queue, offsetStubsDebugHostView, offsetStubsDebugDeviceView);
    alpaka::wait(queue);
    std::cout << "Merge complete. offsetStubs = " << offsetStubsDebug << '\n';
#endif

    // Update the cached information and emit. Both cached scalars are HOST-KNOWN here, so there is
    // nothing to read back:
    //   - offsetBPIX2 is the pixel collection's own scalar, copied verbatim into the output by the
    //     scalar branch of copyColumns() in step 1; the pixel producer set both its device scalar and
    //     its host-side cache from clusters.offsetBPIX2() in the same constructor, so they agree by
    //     construction (TrackingRecHitsDevice.h, "Constructor from clusters");
    //   - offsetStubs is nPixHits -- this module wrote exactly that value into the device scalar in
    //     step 2 above.
    // The previous updateFromDevice(queue) copied the two words back D2H into plain class members,
    // i.e. into PAGEABLE host memory, which turns cudaMemcpyAsync into a synchronising call: two
    // full stream drains per event in a module that contains no alpaka::wait at all. Those two
    // implicit syncs were the whole reason this module inflates x4.55 when the device is contended.
    output.setOffsets(pixRecHitsColl.offsetBPIX2(), nPixHits);
    iEvent.emplace(outputRecHitsSoAToken_, std::move(output));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiPixelRecHitsStubsMerger);
