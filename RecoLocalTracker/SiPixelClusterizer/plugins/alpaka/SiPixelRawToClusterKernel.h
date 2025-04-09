#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelRawToClusterKernel_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelRawToClusterKernel_h

#include <algorithm>
#include <optional>
#include <utility>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigiErrorsSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLTLayout.h"
#include "CondFormats/SiPixelObjects/interface/alpaka/SiPixelGainCalibrationForHLTDevice.h"
#include "CondFormats/SiPixelObjects/interface/alpaka/SiPixelMappingDevice.h"

#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelImageSoA.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelImageDevice.h"

namespace pixelDetails {

  constexpr auto MAX_LINK = pixelgpudetails::MAX_LINK;
  constexpr auto MAX_SIZE = pixelgpudetails::MAX_SIZE;
  constexpr auto MAX_ROC = pixelgpudetails::MAX_ROC;

  // Phase 1 geometry constants
  constexpr uint32_t layerStartBit = 20;
  constexpr uint32_t ladderStartBit = 12;
  constexpr uint32_t moduleStartBit = 2;

  constexpr uint32_t panelStartBit = 10;
  constexpr uint32_t diskStartBit = 18;
  constexpr uint32_t bladeStartBit = 12;

  constexpr uint32_t layerMask = 0xF;
  constexpr uint32_t ladderMask = 0xFF;
  constexpr uint32_t moduleMask = 0x3FF;
  constexpr uint32_t panelMask = 0x3;
  constexpr uint32_t diskMask = 0xF;
  constexpr uint32_t bladeMask = 0x3F;

  constexpr uint32_t LINK_bits = 6;
  constexpr uint32_t ROC_bits = 5;
  constexpr uint32_t DCOL_bits = 5;
  constexpr uint32_t PXID_bits = 8;
  constexpr uint32_t ADC_bits = 8;

  // special for layer 1
  constexpr uint32_t LINK_bits_l1 = 6;
  constexpr uint32_t ROC_bits_l1 = 5;
  constexpr uint32_t COL_bits_l1 = 6;
  constexpr uint32_t ROW_bits_l1 = 7;
  constexpr uint32_t OMIT_ERR_bits = 1;

  constexpr uint32_t maxROCIndex = 8;
  constexpr uint32_t numRowsInRoc = 80;
  constexpr uint32_t numColsInRoc = 52;

  constexpr uint32_t MAX_WORD = 2000;

  constexpr uint32_t ADC_shift = 0;
  constexpr uint32_t PXID_shift = ADC_shift + ADC_bits;
  constexpr uint32_t DCOL_shift = PXID_shift + PXID_bits;
  constexpr uint32_t ROC_shift = DCOL_shift + DCOL_bits;
  constexpr uint32_t LINK_shift = ROC_shift + ROC_bits_l1;
  // special for layer 1 ROC
  constexpr uint32_t ROW_shift = ADC_shift + ADC_bits;
  constexpr uint32_t COL_shift = ROW_shift + ROW_bits_l1;
  constexpr uint32_t OMIT_ERR_shift = 20;

  constexpr uint32_t LINK_mask = ~(~uint32_t(0) << LINK_bits_l1);
  constexpr uint32_t ROC_mask = ~(~uint32_t(0) << ROC_bits_l1);
  constexpr uint32_t COL_mask = ~(~uint32_t(0) << COL_bits_l1);
  constexpr uint32_t ROW_mask = ~(~uint32_t(0) << ROW_bits_l1);
  constexpr uint32_t DCOL_mask = ~(~uint32_t(0) << DCOL_bits);
  constexpr uint32_t PXID_mask = ~(~uint32_t(0) << PXID_bits);
  constexpr uint32_t ADC_mask = ~(~uint32_t(0) << ADC_bits);
  constexpr uint32_t ERROR_mask = ~(~uint32_t(0) << ROC_bits_l1);
  constexpr uint32_t OMIT_ERR_mask = ~(~uint32_t(0) << OMIT_ERR_bits);

  struct DetIdGPU {
    uint32_t rawId;
    uint32_t rocInDet;
    uint32_t moduleId;
  };

  struct Pixel {
    uint32_t row;
    uint32_t col;
  };

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr pixelchannelidentifierimpl::Packing packing() {
    return PixelChannelIdentifier::thePacking;
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t pack(uint32_t row,
                                                              uint32_t col,
                                                              uint32_t adc,
                                                              uint32_t flag = 0) {
    constexpr pixelchannelidentifierimpl::Packing thePacking = packing();
    adc = std::min(adc, uint32_t(thePacking.max_adc));

    return (row << thePacking.row_shift) | (col << thePacking.column_shift) | (adc << thePacking.adc_shift);
  }

  constexpr uint32_t pixelToChannel(int row, int col) {
    constexpr pixelchannelidentifierimpl::Packing thePacking = packing();
    return (row << thePacking.column_width) | col;
  }

}  // namespace pixelDetails

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace pixelDetails {

    class WordFedAppender {
    public:
      WordFedAppender(Queue& queue, uint32_t words)
          : word_{cms::alpakatools::make_host_buffer<unsigned int[]>(queue, words)},
            fedId_{cms::alpakatools::make_host_buffer<unsigned char[]>(queue, words)} {};

      void initializeWordFed(int fedId, unsigned int wordCounterGPU, const uint32_t* src, unsigned int length) {
        std::memcpy(word_.data() + wordCounterGPU, src, sizeof(uint32_t) * length);
        std::memset(fedId_.data() + wordCounterGPU / 2, fedId - 1200, length / 2);
      }
      auto word() const { return word_; }
      auto fedId() const { return fedId_; }

    private:
      cms::alpakatools::host_buffer<unsigned int[]> word_;
      cms::alpakatools::host_buffer<unsigned char[]> fedId_;
    };

    template <typename TrackerTraits>
    class SiPixelRawToClusterKernel {
    public:
      SiPixelRawToClusterKernel() : nModules_Clusters_h{cms::alpakatools::make_host_buffer<uint32_t[], Platform>(3u)} {}

      ~SiPixelRawToClusterKernel() = default;

      SiPixelRawToClusterKernel(const SiPixelRawToClusterKernel&) = delete;
      SiPixelRawToClusterKernel(SiPixelRawToClusterKernel&&) = delete;
      SiPixelRawToClusterKernel& operator=(const SiPixelRawToClusterKernel&) = delete;
      SiPixelRawToClusterKernel& operator=(SiPixelRawToClusterKernel&&) = delete;

      void makePhase1ClustersAsync(Queue& queue,
                                   const SiPixelClusterThresholds clusterThresholds,
				   SiPixelImageSoAView images_,
                                   const SiPixelMappingSoAConstView& cablingMap,
                                   const unsigned char* modToUnp,
                                   const SiPixelGainCalibrationForHLTSoAConstView& gains,
                                   const WordFedAppender& wordFed,
                                   const uint32_t wordCounter,
                                   const uint32_t fedCounter,
                                   bool useQualityInfo,
                                   bool includeErrors,
                                   bool debug);

      void makePhase2ClustersAsync(Queue& queue,
                                   const SiPixelClusterThresholds clusterThresholds,
                                   SiPixelDigisSoAView& digis_view,
                                   const uint32_t numDigis);

      SiPixelDigisSoACollection getDigis() {
        digis_d->setNModules(nModules_Clusters_h[0]);
        return std::move(*digis_d);
      }

      SiPixelClustersSoACollection getClusters() {
        clusters_d->setNClusters(nModules_Clusters_h[1], nModules_Clusters_h[2]);
        return std::move(*clusters_d);
      }

      SiPixelDigiErrorsSoACollection getErrors() { return std::move(*digiErrors_d); }

      auto nModules() { return nModules_Clusters_h[0]; }

    private:
      uint32_t nDigis = 0;

      // Data to be put in the event
      cms::alpakatools::host_buffer<uint32_t[]> nModules_Clusters_h;
      std::optional<SiPixelDigisSoACollection> digis_d;
      std::optional<SiPixelClustersSoACollection> clusters_d;
      std::optional<SiPixelDigiErrorsSoACollection> digiErrors_d;
    };

  }  // namespace pixelDetails
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // plugin_SiPixelClusterizer_alpaka_SiPixelRawToClusterKernel_h
