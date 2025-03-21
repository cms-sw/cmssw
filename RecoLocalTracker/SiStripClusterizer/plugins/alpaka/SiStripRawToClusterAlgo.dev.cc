// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"

#include "SiStripRawToClusterAlgo.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"
// #include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"

// kernels and related objects
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;

  // Phase 1 geometry constants

  constexpr uint16_t FED_ID_MIN = sistrip::FED_ID_MIN;
  constexpr uint16_t FEDCH_PER_FED = sistrip::FEDCH_PER_FED;
  constexpr uint16_t STRIPS_PER_FEDCH = sistrip::STRIPS_PER_FEDCH;
  constexpr uint16_t badBit = (1 << 15);
  constexpr uint16_t invalidStrip = std::numeric_limits<uint16_t>::max();
  constexpr uint16_t APVS_PER_FEDCH = sistrip::APVS_PER_FEDCH;
  constexpr uint16_t APVS_PER_CHAN = sistrip::APVS_PER_CHAN;
  constexpr uint16_t STRIPS_PER_APV = sistrip::STRIPS_PER_APV;

  ALPAKA_FN_ACC constexpr int kMaxSeedStrips = 200000;
  constexpr uint16_t stripIndexMask = 0x7FFF;

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint16_t fedIndex(uint16_t fed) { return fed - FED_ID_MIN; }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t stripIndex(uint16_t fedID, uint8_t fedCH, uint16_t strip) {
    return fedIndex(fedID) * FEDCH_PER_FED * STRIPS_PER_FEDCH + fedCH * STRIPS_PER_FEDCH + (strip % STRIPS_PER_FEDCH);
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t apvIndex(uint16_t fed, uint8_t channel, uint16_t strip) {
    return fedIndex(fed) * APVS_PER_FEDCH * FEDCH_PER_FED + APVS_PER_CHAN * channel +
           (strip % STRIPS_PER_FEDCH) / STRIPS_PER_APV;
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t channelIndex(uint16_t fedID, uint8_t fedCH) {
    return fedIndex(fedID) * FEDCH_PER_FED + fedCH;
  }

  class SiStripRawToClusterAlgoKernel_unpacker {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  SiStripMappingConstView mapping,
                                  SiStripClusterizerConditionsDetToFedsConstView DetToFeds,
                                  SiStripClusterizerConditionsData_fedchConstView Data_fedch,
                                  SiStripClusterizerConditionsData_stripConstView Data_strip,
                                  SiStripClusterizerConditionsData_apvConstView Data_apv,
                                  StripDigiView stripDataObj) const {
      // // set this only once in the whole kernel grid
      // if (once_per_grid(acc)) {
      //   stripDataObj.adc(0) = 1;
      // }
      // return;

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (auto chan : uniform_elements(acc, mapping.metadata().size())) {
        const auto fedID = mapping.fedID(chan);
        const auto fedCH = mapping.fedCh(chan);

        const auto ipair = Data_fedch.iPair_(channelIndex(fedID, fedCH));
        const auto ipoff = STRIPS_PER_FEDCH * ipair;

        const auto data = mapping.input(chan);
        const auto len = mapping.length(chan);

        if (data != nullptr && len > 0) {
          auto aoff = mapping.offset(chan);
          auto choff = mapping.inoff(chan);
          const auto end = choff + len;

          while (choff < end) {
            auto stripIndex = data[(choff++) ^ 7] + ipoff;
            const auto groupLength = data[(choff++) ^ 7];

            // initialize as invalid strip
            for (auto i = 0; i < 2; ++i) {
              stripDataObj.stripId(aoff) = 0xFFFF;
              stripDataObj.adc(aoff++) = 0;
            }

            for (auto i = 0; i < groupLength; ++i) {
              stripDataObj.stripId(aoff) = stripIndex++;
              stripDataObj.channel(aoff) = chan;
              stripDataObj.adc(aoff++) = data[(choff++) ^ 7];
            }
          }
        }  // choff < end
      }  // data != nullptr && len > 0
    }
  };

  class SiStripRawToClusterAlgoKernel_setSeedStripsGPU {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  SiStripMappingConstView mapping,
                                  SiStripClusterizerConditionsData_stripConstView Data_strip,
                                  StripDigiConstView stripDataObj,
                                  StripClustersAuxView clusterDataObj) const {
      // make a strided loop over the kernel grid, covering up to "size" elements
      auto nStrips = stripDataObj.metadata().size();
      const float seedThreshold = clusterDataObj.seedThreshold();
      for (auto chan : uniform_elements(acc, nStrips)) {
        clusterDataObj.seedStripsMask(chan) = 0;
        clusterDataObj.seedStripsNCMask(chan) = 0;
        const auto stripID = stripDataObj.stripId(chan);
        if (stripID != invalidStrip) {
          const auto chan_ = stripDataObj.channel(chan);
          const auto fedID = mapping.fedID(chan_);
          const auto fedCH = mapping.fedCh(chan_);

          const auto idx = stripIndex(fedID, fedCH, stripID);
          uint16_t noise_tmp = Data_strip.noise_(idx);

          const float noise_i = 0.1f * (noise_tmp & ~badBit);
          const uint8_t adc_i = stripDataObj.adc(chan);

          clusterDataObj.seedStripsMask(chan) = (adc_i >= static_cast<uint8_t>(noise_i * seedThreshold)) ? 1 : 0;
          clusterDataObj.seedStripsNCMask(chan) = clusterDataObj.seedStripsMask(chan);
          // clusterDataObj.seedStripsNCMask(chan) = static_cast<uint8_t>(seedThreshold);
        }
      }
    }
  };

  class SiStripRawToClusterAlgoKernel_setNCSeedStripsGPU {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  SiStripMappingConstView mapping,
                                  StripDigiConstView stripDataObj,
                                  StripClustersAuxView clusterDataObj) const {
      auto channels = stripDataObj.channel();
      // make a strided loop over the kernel grid, covering up to "size" elements
      for (auto stripIdx : uniform_elements(acc, 1, stripDataObj.metadata().size())) {
        const auto detid = mapping.detID(channels[stripIdx]);
        const auto detid1 = mapping.detID(channels[stripIdx - 1]);

        if (clusterDataObj.seedStripsMask(stripIdx) && clusterDataObj.seedStripsMask(stripIdx - 1) &&
            (stripDataObj.stripId(stripIdx) - stripDataObj.stripId(stripIdx - 1)) == 1 && (detid == detid1)) {
          clusterDataObj.seedStripsNCMask(stripIdx) = 0;
        }
      }
    }
  };

  class SiStripRawToClusterAlgoKernel_setStripIndex {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  StripDigiConstView stripDataObj,
                                  StripClustersAuxView clusterDataObj) const {
      // make a strided loop over the kernel grid, covering up to "size" elements
      for (auto stripIdx : uniform_elements(acc, stripDataObj.metadata().size())) {
        if (clusterDataObj.seedStripsNCMask(stripIdx) == 1) {
          const int index = (clusterDataObj.prefixSeedStripsNCMask(stripIdx) - 1);
          clusterDataObj.seedStripsNCIndex(index) = stripIdx;
        }
      }
    }
  };

  class SiStripRawToClusterAlgoKernel_findLeftRightBoundaryGPU {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  SiStripMappingConstView mapping,
                                  SiStripClusterizerConditionsData_stripConstView Data_strip,
                                  StripDigiConstView stripDataObj,
                                  StripClustersAuxView clusterDataObj,
                                  sistrip::SiStripClustersView clusters) const {
      const auto nStrips = stripDataObj.metadata().size();
      const auto nSeedStripsNC = std::min(kMaxSeedStrips, clusterDataObj.prefixSeedStripsNCMask(nStrips - 1));
      auto channels = stripDataObj.channel();
      auto stripId = stripDataObj.stripId();
      auto adc = stripDataObj.adc();
      const auto channelThreshold = clusterDataObj.channelThreshold();
      const auto maxSequentialHoles = clusterDataObj.maxSequentialHoles();
      int clusterSizeLimit = clusterDataObj.clusterSizeLimit();
      const auto clusterThresholdSquared = clusterDataObj.clusterThresholdSquared();

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (auto i : uniform_elements(acc, nSeedStripsNC)) {
        const auto index = clusterDataObj.seedStripsNCIndex(i);
        const auto chan = channels[index];
        const auto fed = mapping.fedID(chan);
        const auto channel = mapping.fedCh(chan);
        const auto det = mapping.detID(chan);
        const auto strip = stripId[index];
        //
        const auto idx = stripIndex(fed, channel, strip);
        uint16_t noise_tmp = Data_strip.noise_(idx);
        const float noise_i = 0.1f * (noise_tmp & ~badBit);

        auto noiseSquared_i = noise_i * noise_i;
        float adcSum_i = static_cast<float>(adc[index]);
        auto testIndex = index - 1;
        int size = 1;

        auto addtocluster = [&](int& indexLR) {
          const auto testchan = channels[testIndex];
          const auto testFed = mapping.fedID(testchan);
          const auto testChannel = mapping.fedCh(testchan);
          const auto testStrip = stripId[testIndex];

          const auto idx = stripIndex(testFed, testChannel, testStrip);
          uint16_t noise_tmp = Data_strip.noise_(idx);
          const float testNoise = 0.1f * (noise_tmp & ~badBit);

          const auto testADC = adc[testIndex];

          if (testADC >= static_cast<uint8_t>(testNoise * channelThreshold)) {
            ++size;
            indexLR = testIndex;
            noiseSquared_i += testNoise * testNoise;
            adcSum_i += static_cast<float>(testADC);
          }
        };

        // find left boundary
        auto indexLeft = index;

        if (testIndex >= 0 && stripId[testIndex] == invalidStrip) {
          testIndex -= 2;
        }

        if (testIndex >= 0) {
          const auto testchan = channels[testIndex];
          const auto testDet = mapping.detID(testchan);
          auto rangeLeft = stripId[indexLeft] - stripId[testIndex] - 1;
          auto sameDetLeft = det == testDet;

          while (sameDetLeft && (rangeLeft >= 0) && (rangeLeft <= maxSequentialHoles) &&
                 (size < (clusterSizeLimit + 1))) {
            addtocluster(indexLeft);
            --testIndex;
            if (testIndex >= 0 && stripId[testIndex] == invalidStrip) {
              testIndex -= 2;
            }
            if (testIndex >= 0) {
              rangeLeft = stripId[indexLeft] - stripId[testIndex] - 1;
              const auto newchan = channels[testIndex];
              const auto newdet = mapping.detID(newchan);
              sameDetLeft = det == newdet;
            } else {
              sameDetLeft = false;
            }
          }  // while loop
        }  // testIndex >= 0

        // find right boundary
        auto indexRight = index;
        testIndex = index + 1;

        if (testIndex < nStrips && stripId[testIndex] == invalidStrip) {
          testIndex += 2;
        }

        if (testIndex < nStrips) {
          const auto testchan = channels[testIndex];
          const auto testDet = mapping.detID(testchan);
          auto rangeRight = stripId[testIndex] - stripId[indexRight] - 1;
          auto sameDetRight = det == testDet;

          while (sameDetRight && (rangeRight >= 0) && (rangeRight <= maxSequentialHoles) &&
                 (size < (clusterSizeLimit + 1))) {
            addtocluster(indexRight);
            ++testIndex;
            if (testIndex < nStrips && stripId[testIndex] == invalidStrip) {
              testIndex += 2;
            }
            if (testIndex < nStrips) {
              rangeRight = stripId[testIndex] - stripId[indexRight] - 1;
              const auto newchan = channels[testIndex];
              const auto newdet = mapping.detID(newchan);
              sameDetRight = det == newdet;
            } else {
              sameDetRight = false;
            }
          }  // while loop
        }  // testIndex < nStrips
        clusters.clusterIndex(i) = indexLeft;
        clusters.clusterSize(i) = indexRight - indexLeft + 1;
        clusters.clusterDetId(i) = det;
        clusters.firstStrip(i) = stripId[indexLeft];
        clusters.trueCluster(i) = (noiseSquared_i * clusterThresholdSquared <= adcSum_i * adcSum_i) and
                                  (clusters.clusterSize(i) <= static_cast<uint32_t>(clusterSizeLimit));
      }  // i < nSeedStripsNC

      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        clusters.nClusters() = nSeedStripsNC;
        clusters.maxClusterSize() = clusterSizeLimit;
      }
    }
  };

  class SiStripRawToClusterAlgoKernel_checkClusterConditionGPU {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  SiStripMappingConstView mapping,
                                  SiStripClusterizerConditionsData_fedchConstView Data_fedch,
                                  SiStripClusterizerConditionsData_apvConstView Data_apv,
                                  StripDigiConstView stripDataObj,
                                  StripClustersAuxView clusterDataObj,
                                  sistrip::SiStripClustersView clusters) const {
      const auto nSeedStripsNC = clusters.nClusters();
      auto trueCluster = clusters.trueCluster();
      auto clusterIndexLeft = clusters.clusterIndex();
      auto clusterSize = clusters.clusterSize();
      auto clusterADCs = clusters.clusterADCs();
      auto adc = stripDataObj.adc();
      auto charge = clusters.charge();
      auto barycenter = clusters.barycenter();
      auto channels = stripDataObj.channel();
      auto stripId = stripDataObj.stripId();
      auto minGoodCharge = clusterDataObj.minGoodCharge();
      constexpr uint8_t adc_low_saturation = 254;
      constexpr uint8_t adc_high_saturation = 255;
      constexpr int charge_low_saturation = 253;
      constexpr int charge_high_saturation = 1022;

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (auto i : uniform_elements(acc, nSeedStripsNC)) {
        if (trueCluster[i]) {
          unsigned int left = clusterIndexLeft[i];
          unsigned int size = clusterSize[i];

          if (i > 0 && clusterIndexLeft[i - 1] == left) {
            trueCluster[i] = 0;  // ignore duplicates
          } else {
            float adcSum = 0.0f;
            int sumx = 0;
            int suma = 0;

            int j = 0;
            for (unsigned int k = 0; k < size; k++) {
              auto index = left + k;
              auto chan = channels[index];
              auto fed = mapping.fedID(chan);
              auto channel = mapping.fedCh(chan);
              auto strip = stripId[index];

              if (strip != invalidStrip) {
                float gain_j = Data_apv.gain_(apvIndex(fed, channel, strip));

                uint8_t adc_j = adc[index];
                const int charge = static_cast<int>(static_cast<float>(adc_j) / gain_j + 0.5f);

                if (adc_j < adc_low_saturation) {
                  adc_j = (charge > charge_high_saturation
                               ? adc_high_saturation
                               : (charge > charge_low_saturation ? adc_low_saturation : charge));
                }
                clusterADCs[i][j] = adc_j;

                adcSum += static_cast<float>(adc_j);
                sumx += j * adc_j;
                suma += adc_j;
                j++;
              }
            }  // loop over cluster strips
            charge[i] = adcSum;
            auto chan = channels[left];
            auto fed = mapping.fedID(chan);
            auto channel = mapping.fedCh(chan);
            trueCluster[i] = (adcSum * Data_fedch.invthick_(channelIndex(fed, channel))) > minGoodCharge;
            auto bary_i = static_cast<float>(sumx) / static_cast<float>(suma);
            barycenter[i] = static_cast<float>(stripId[left] & stripIndexMask) + bary_i + 0.5f;
            clusterSize[i] = j;
          }  // not a duplicate cluster
        }  // trueCluster[i] is true
      }  // i < nSeedStripsNC
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// kernels launchers
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  using namespace sistripclusterizer;

  SiStripRawToClusterAlgo::SiStripRawToClusterAlgo(const edm::ParameterSet& conf)
      : channelThreshold_(conf.getParameter<double>("ChannelThreshold")),
        seedThreshold_(conf.getParameter<double>("SeedThreshold")),
        clusterThresholdSquared_(std::pow(conf.getParameter<double>("ClusterThreshold"), 2.0f)),
        maxSequentialHoles_(conf.getParameter<unsigned>("MaxSequentialHoles")),
        maxSequentialBad_(conf.getParameter<unsigned>("MaxSequentialBad")),
        maxAdjacentBad_(conf.getParameter<unsigned>("MaxAdjacentBad")),
        maxClusterSize_(conf.getParameter<unsigned>("MaxClusterSize")),
        minGoodCharge_(clusterChargeCut(conf)) {}

  void SiStripRawToClusterAlgo::initialize(Queue& queue, int n_strips) {
    // Create unpakcedStrips on host and initialize with algo parameters
    auto sClustersAux_host = StripClusterizerHost({{n_strips, n_strips}}, queue);
    sClustersAux_host.zeroInitialise();

    // Initialize the members of the clusterizer
    auto stripDigi_host = sClustersAux_host.view<StripClustersAuxSoA>();
    stripDigi_host.channelThreshold() = channelThreshold_;
    stripDigi_host.seedThreshold() = seedThreshold_;
    stripDigi_host.clusterThresholdSquared() = clusterThresholdSquared_;
    stripDigi_host.maxSequentialHoles() = maxSequentialHoles_;
    stripDigi_host.maxSequentialBad() = maxSequentialBad_;
    stripDigi_host.maxAdjacentBad() = maxAdjacentBad_;
    stripDigi_host.minGoodCharge() = minGoodCharge_;
    stripDigi_host.clusterSizeLimit() = maxClusterSize_;

    clustersAux_d_ = StripClusterizerDevice({{n_strips, n_strips}}, queue);
    alpaka::memcpy(queue, clustersAux_d_->buffer(), sClustersAux_host.const_buffer());
  }

  void SiStripRawToClusterAlgo::unpackStrips(Queue& queue,
                                             SiStripMappingDevice const& mapping,
                                             SiStripClusterizerConditionsDevice const& conditions) {
    // In HeterogeneousCore/AlpakaTest, typical sizes are power of 2 like 32 and 64.
    // The hw number of threads in nvidia devices is max 1024. The number of strips is up to
    // (sistrip::STRIPS_PER_FED = 24576) * (sistrip::NUMBER_OF_FEDS = ) => 24576*440 = 10813440
    // Assume conditions cut this to 80%, then 10813440*0.8 = 8650752
    uint32_t threads =
        512;  // I wonder if there is an helper function which automatically optimize this based on the accelerator properties
    // use as many groups as needed to cover the whole problem
    auto nStrips = mapping->metadata().size();
    uint32_t groups = divide_up_by(nStrips, threads);

    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(groups, threads);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripRawToClusterAlgoKernel_unpacker{},
                        mapping.const_view(),
                        conditions.const_view<SiStripClusterizerConditionsDetToFedsSoA>(),
                        conditions.const_view<SiStripClusterizerConditionsData_fedchSoA>(),
                        conditions.const_view<SiStripClusterizerConditionsData_stripSoA>(),
                        conditions.const_view<SiStripClusterizerConditionsData_apvSoA>(),
                        clustersAux_d_->view());

#ifdef EDM_ML_DEBUG
    auto sClustersAux_host = StripClusterizerHost(clustersAux_d_->sizes(), queue);
    alpaka::memcpy(queue, sClustersAux_host.buffer(), clustersAux_d_->buffer());
    alpaka::wait(queue);
    LogDebug("SiStripUnpkDigi") << "[SiStripRawToClusterAlgo::unpackStrips] Dumping channel_...\n"
                                << "i\tadc\tchan\tstripId\n";
    for (int i = 0; i < sClustersAux_host->metadata().size(); ++i) {
      if (i % 100 == 0 and sClustersAux_host->stripId(i) != invalidStrip) {
        LogDebug("SiStripUnpkDigi") << i << " : " << (int)sClustersAux_host->adc(i) << " "
                                    << (int)sClustersAux_host->channel(i) << " " << (int)sClustersAux_host->stripId(i)
                                    << "\n";
      }
    }
#endif
  }

  void SiStripRawToClusterAlgo::setSeedsAndMakeIndexes(Queue& queue,
                                                       SiStripMappingDevice const& mapping,
                                                       SiStripClusterizerConditionsDevice const& conditions) {
    // In HeterogeneousCore/AlpakaTest, typical sizes are power of 2 like 32 and 64.
    // I wonder if there is an helper function which automatically optimize this based on the accelerator properties.
    // Most likely I could retrieve the device attached to the queue (alpaka::getDev(queue)) and then depending on its properties set the optimal threads
    uint32_t threads = 512;
    auto nStrips = mapping->metadata().size();
    uint32_t groups = divide_up_by(nStrips, threads);

    auto workDiv = make_workdiv<Acc1D>(groups, threads);

    // Set the seeds according to noise and seedThreshold
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripRawToClusterAlgoKernel_setSeedStripsGPU{},
                        mapping.const_view(),
                        conditions.const_view<SiStripClusterizerConditionsData_stripSoA>(),
                        clustersAux_d_->const_view(),
                        clustersAux_d_->view<StripClustersAuxSoA>());

#if defined(EDM_ML_DEBUG) && defined(SUPERDETAILS)
    alpaka::wait(queue);
    checkUnpackedStrips_(queue, clustersAux_d_.value());
#endif

    // Flag the non-contiguous strips (in the same detector) with 0
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripRawToClusterAlgoKernel_setNCSeedStripsGPU{},
                        mapping.const_view(),
                        clustersAux_d_->const_view(),
                        clustersAux_d_->view<StripClustersAuxSoA>());

    // Calculate the prefix for the non-contiguous flagged strips and store in prefixSeedStripsNCMask
    uint32_t num_items = clustersAux_d_->view().metadata().size();
    const auto nThreads = 1024;
    int32_t nBlocks = divide_up_by(num_items, nThreads);
    auto workDivMultiBlock = make_workdiv<Acc1D>(nBlocks, nThreads);
    auto blockCounter_d = make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, blockCounter_d, 0);
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc1D>(workDivMultiBlock,
                                        multiBlockPrefixScan<int>(),
                                        clustersAux_d_->const_view<StripClustersAuxSoA>().seedStripsNCMask(),
                                        clustersAux_d_->view<StripClustersAuxSoA>().prefixSeedStripsNCMask(),
                                        num_items,
                                        nBlocks,
                                        blockCounter_d.data(),
                                        alpaka::getPreferredWarpSize(alpaka::getDev(queue))));

#if defined(EDM_ML_DEBUG) && defined(SUPERDETAILS)
    alpaka::wait(queue);
    checkPrefixSum_(queue, clustersAux_d_.value());
#endif

    // Attach to the index according to the *exclusive* prefix sum when contiguous strips are found
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripRawToClusterAlgoKernel_setStripIndex{},
                        clustersAux_d_->const_view(),
                        clustersAux_d_->view<StripClustersAuxSoA>());

#ifdef EDM_ML_DEBUG
    checkUnpackedStrips_(queue, clustersAux_d_.value());
#endif
  }

  void SiStripRawToClusterAlgo::makeClusters(Queue& queue,
                                             SiStripMappingDevice const& mapping,
                                             SiStripClusterizerConditionsDevice const& conditions) {
    // The maximum number of clusters is set to kMaxSeedStrips
    clusters_d_ = sistrip::SiStripClustersDevice(kMaxSeedStrips, queue);
    // The number of seed over which to loop for clusters is the min between the number of strips and the kMaxSeeds
    const auto nStrips = clustersAux_d_->view().metadata().size();
    const int nSeeds = std::min(kMaxSeedStrips, nStrips);

    // Create a work division with not-so large thread number, as the clustering runs over chunks of memory and typical cluster size is limited to about 16
    uint32_t threads = 512;
    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(nSeeds, threads);
    auto workDiv = make_workdiv<Acc1D>(groups, threads);

    // Three-threshold clusterization algo
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripRawToClusterAlgoKernel_findLeftRightBoundaryGPU{},
                        mapping.const_view(),
                        conditions.const_view<SiStripClusterizerConditionsData_stripSoA>(),
                        clustersAux_d_->const_view(),
                        clustersAux_d_->view<StripClustersAuxSoA>(),
                        clusters_d_->view());

    // Apply the conditions
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripRawToClusterAlgoKernel_checkClusterConditionGPU{},
                        mapping.const_view(),
                        conditions.const_view<SiStripClusterizerConditionsData_fedchSoA>(),
                        conditions.const_view<SiStripClusterizerConditionsData_apvSoA>(),
                        clustersAux_d_->const_view(),
                        clustersAux_d_->view<StripClustersAuxSoA>(),
                        clusters_d_->view());

#if defined(EDM_ML_DEBUG) && defined(SUPERDETAILS)
    checkClusters_(queue);
#endif
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#ifdef EDM_ML_DEBUG  // debug functions
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  void SiStripRawToClusterAlgo::checkUnpackedStrips_(Queue& queue, StripClusterizerDevice& output) const {
    auto output_onHost = StripClusterizerHost(output.sizes(), queue);
    alpaka::memcpy(queue, output_onHost.buffer(), output.buffer());
    alpaka::wait(queue);

    int size = output->metadata().size();

    LogDebug("SiStripUnpkDigi") << " Size of StripDigiDevice " << size << "\n";
    LogDebug("SiStripUnpkDigi")
        << "i\tadc\tchannel\tstripId\tseedStripsMask\tseedStripsNCMask\tprefixSeedStripsNCMask\tseedStripsNCIndex\n";
    for (int i = 0; i < size; ++i) {
      if ((i < 100 || i % 1000 == 0) && output_onHost->stripId(i) != invalidStrip) {
        LogDebug("SiStripUnpkDigi") << i << "\t" << (int)output_onHost->adc(i) << "\t" << output_onHost->channel(i)
                                    << "\t" << output_onHost->stripId(i) << "\t"
                                    << output_onHost.view<StripClustersAuxSoA>().seedStripsMask(i) << "\t"
                                    << output_onHost.view<StripClustersAuxSoA>().seedStripsNCMask(i) << "\t"
                                    << output_onHost.view<StripClustersAuxSoA>().prefixSeedStripsNCMask(i) << "\t"
                                    << output_onHost.view<StripClustersAuxSoA>().seedStripsNCIndex(i) << "\n";
      }
    }
    alpaka::wait(queue);
  }

  void SiStripRawToClusterAlgo::checkPrefixSum_(Queue& queue, StripClusterizerDevice& output) const {
    auto output_onHost = StripClusterizerHost(output.sizes(), queue);
    alpaka::memcpy(queue, output_onHost.buffer(), output.buffer());
    alpaka::wait(queue);

    auto seedStripsNCMask = output_onHost.view<StripClustersAuxSoA>().seedStripsNCMask();
    auto prefixSeedStripsNCMask = output_onHost.view<StripClustersAuxSoA>().prefixSeedStripsNCMask();

    int size = output->metadata().size();
    LogDebug("SiStripUnpkDigi") << "[SiStripRawToClusterAlgo::checkPrefixSum_] Size of seedStripsNCMask " << size
                                << "\n";
    LogDebug("SiStripUnpkDigi") << "i\tseedStripsNCMask\tprefixSeedStripsNCMask\n";
    for (int i = 0; i < size; ++i) {
      LogDebug("SiStripUnpkDigi") << i << "\t" << seedStripsNCMask[i] << "\t" << prefixSeedStripsNCMask[i] << "\n";
    }
  }

  void SiStripRawToClusterAlgo::checkClusters_(Queue& queue) const {
    // auto clusters_device = sistrip::SiStripClustersDevice(clusters->metadata().size(), queue);
    // fill(queue, clusters_device, 2.3);

    // Copy the clusted on the host
    auto clusters_host = sistrip::SiStripClustersHost(clusters_d_->view().metadata().size(), queue);
    alpaka::memcpy(queue, clusters_host.buffer(), clusters_d_->buffer());
    alpaka::wait(queue);

    LogDebug("SiStripClusChk") << "clusters->metadata().size() = " << clusters_d_->view().metadata().size() << "\n";
    LogDebug("SiStripClusChk") << "nClusters\tmaxClusterSize\n";
    LogDebug("SiStripClusChk") << clusters_host->nClusters() << "\t" << clusters_host->maxClusterSize() << "\n";
    LogDebug("SiStripClusChk") << "   -----  ---------- -----   \n";
    LogDebug("SiStripClusChk") << "i\tcIdx\tcSz\tcDetId\t1Strip\ttrueC\tbary\tchg\t - clusterADCs\n";
    // Print the result
    for (int i = 0; i < (int)clusters_host->nClusters(); ++i) {
      if (i > 100 && i < ((int)clusters_host->nClusters() - 1000))
        continue;
      LogDebug("SiStripClusChk") << i << "\t" << clusters_host->clusterIndex(i) << "\t" << clusters_host->clusterSize(i)
                                 << "\t" << clusters_host->clusterDetId(i) << "\t" << clusters_host->charge(i) << "\t"
                                 << clusters_host->firstStrip(i) << "\t" << clusters_host->trueCluster(i) << "\t"
                                 << clusters_host->barycenter(i) << "\t - ";
      for (unsigned int j = 0; j < clusters_host->clusterSize(i); ++j) {
        LogDebug("") << j << ":" << (int)(clusters_host->clusterADCs(i)[j]) << "  ";
      }
      LogDebug("") << "\n";
    }
    alpaka::wait(queue);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif  // debug functions