#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/warpsize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/moveToDeviceAsync.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsStruct.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"

#include "SiStripRawToClusterAlgo.h"

// Generic raw unpackers

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip::fedchannelunpacker {
  enum class StatusCode { SUCCESS = 0, BAD_CHANNEL_LENGTH, UNORDERED_DATA, BAD_PACKET_CODE, ZERO_PACKET_CODE };
  namespace detail {
    template <uint8_t num_words>
    ALPAKA_FN_HOST_ACC uint16_t getADC_W(const uint8_t* data, uint_fast16_t offset, uint8_t bits_shift) {
      // get ADC from one or two bytes (at most 10 bits), and shift if needed
      return (data[offset ^ 7] + (num_words == 2 ? ((data[(offset + 1) ^ 7] & 0x03) << 8) : 0)) << bits_shift;
    }

    template <uint16_t mask>
    ALPAKA_FN_HOST_ACC uint16_t getADC_B2(const uint8_t* data, uint_fast16_t wOffset, uint_fast8_t bOffset) {
      // get ADC from two bytes, from wOffset until bOffset bits from the next byte (maximum decided by mask)
      return (((data[wOffset ^ 7]) << bOffset) + (data[(wOffset + 1) ^ 7] >> (BITS_PER_BYTE - bOffset))) & mask;
    }

    template <uint16_t mask>
    ALPAKA_FN_HOST_ACC uint16_t getADC_B1(const uint8_t* data, uint_fast16_t wOffset, uint_fast8_t bOffset) {
      // get ADC from one byte, until bOffset into the byte at wOffset (maximum decided by mask)
      return (data[wOffset ^ 7] >> (BITS_PER_BYTE - bOffset)) & mask;
    }

    template <uint8_t num_bits>
    ALPAKA_FN_HOST_ACC StatusCode unpackZSW(uint32_t chan,
                                            const uint8_t* channel_data,
                                            uint16_t channel_length,
                                            uint32_t channel_offset,
                                            SiStripDigiView out,
                                            uint32_t* aoffIdx,
                                            uint8_t headerLength,
                                            uint16_t stripStart,
                                            uint8_t bits_shift = 0) {
      constexpr auto num_words = num_bits / 8;
      static_assert(((num_bits % 8) == 0) && (num_words > 0) && (num_words < 3));
      if (channel_length & 0xF000) {
        printf("FEDBuffer | Channel length is invalid. Channel length is %i\n", channel_length);
        return StatusCode::BAD_CHANNEL_LENGTH;
      }
      const uint8_t* const data = channel_data;
      uint_fast16_t offset = channel_offset + headerLength;  // header is 2 (lite) or 7
      uint_fast8_t firstStrip{0}, nInCluster{0}, inCluster{0};
      const uint_fast16_t end = channel_offset + channel_length;
      while (offset != end) {
        if (inCluster == nInCluster) {
          if (offset + 2 >= end) {
            // offset should already be at end then (empty cluster)
            // printf("offset should already set");
            break;
          }
          const uint_fast8_t newFirstStrip = data[(offset++) ^ 7];
          if (newFirstStrip < (firstStrip + inCluster)) {
            // LogDebug("FEDBuffer") << "First strip of new cluster is not greater than last strip of previous cluster. "
            //                       << "Last strip of previous cluster is " << uint16_t(firstStrip + inCluster) << ". "
            //                       << "First strip of new cluster is " << uint16_t(newFirstStrip) << ".";
            // printf("StatusCode::UNORDERED_DATA");
            return StatusCode::UNORDERED_DATA;
          }
          firstStrip = newFirstStrip;
          nInCluster = data[(offset++) ^ 7];
          inCluster = 0;
          // out.stripId(++(*aoffIdx)) = 0xFFFF; // wonder what is the reason for these offsets
          // from my investigation, there is always a call at the begin and end of the cluster.
          // If this is a general property, then these two rows of the out StripDigiView could be saved for each unpackZSW call
          // for (int i = 0; i < 2; ++i, ++(*aoffIdx)) {
          //   out.stripId(*aoffIdx) = 0xFFFF;
          //   out.adc(*aoffIdx) = 0;
          // }
        }
        // assert(*aoffIdx!=23);
        out.channel(*aoffIdx) = chan;
        out.stripId(*aoffIdx) = stripStart + firstStrip + inCluster;
        out.adc(*aoffIdx) = getADC_W<num_words>(data, offset, bits_shift);
        (*aoffIdx)++;
        offset += num_words;
        ++inCluster;
      }
      return StatusCode::SUCCESS;
    }

    // Generic implementation (for 10bit, essentially)
    template <uint_fast8_t num_bits>
    ALPAKA_FN_HOST_ACC StatusCode unpackZSB(uint32_t chan,
                                            const uint8_t* channel_data,
                                            uint16_t channel_length,
                                            uint32_t channel_offset,
                                            SiStripDigiView out,
                                            uint32_t* idx,
                                            uint8_t headerLength,
                                            uint16_t stripStart) {
      constexpr uint16_t mask = (1 << num_bits) - 1;
      if (channel_length & 0xF000) {
        // LogDebug("FEDBuffer") << "Channel length is invalid. Channel length is " << uint16_t(channel.length()) << ".";
        // printf("[%i] | BAD_CHANNEL_LENGTH\n", *idx);
        return StatusCode::BAD_CHANNEL_LENGTH;
      }
      uint_fast16_t wOffset = channel_offset + headerLength;  // header is 2 (lite) or 7
      uint_fast8_t bOffset{0}, firstStrip{0}, nInCluster{0}, inCluster{0};
      const uint_fast16_t chEnd = channel_offset + channel_length;
      while (((wOffset + 1) < chEnd) ||
             ((inCluster != nInCluster) && ((chEnd - wOffset) * BITS_PER_BYTE - bOffset >= num_bits))) {
        if (inCluster == nInCluster) {
          if (wOffset + 2 >= chEnd) {
            // offset should already be at end then (empty cluster)
            break;
          }
          if (bOffset) {
            ++wOffset;
            bOffset = 0;
          }
          const uint_fast8_t newFirstStrip = channel_data[(wOffset++) ^ 7];
          if (newFirstStrip < (firstStrip + inCluster)) {
            // LogDebug("FEDBuffer") << "First strip of new cluster is not greater than last strip of previous cluster. "
            //                       << "Last strip of previous cluster is " << uint16_t(firstStrip + inCluster) << ". "
            //                       << "First strip of new cluster is " << uint16_t(newFirstStrip) << ".";
            // printf("[%i] | UNORDERED_DATA - %i\t%i\t%i\n", *idx, newFirstStrip, firstStrip, inCluster);
            return StatusCode::UNORDERED_DATA;
          }
          firstStrip = newFirstStrip;
          nInCluster = channel_data[(wOffset++) ^ 7];
          inCluster = 0;
          bOffset = 0;
        }
        bOffset += num_bits;
        if ((num_bits > BITS_PER_BYTE) || (bOffset > BITS_PER_BYTE)) {
          bOffset -= BITS_PER_BYTE;
          out.adc(*idx) = getADC_B2<mask>(channel_data, wOffset, bOffset);
          (*idx)++;
          ++wOffset;
        } else {
          out.adc(*idx) = getADC_B1<mask>(channel_data, wOffset, bOffset);
          (*idx)++;
        }
        out.channel(*idx) = chan;
        out.stripId(*idx) = stripStart + firstStrip + inCluster;
        ++inCluster;
        if (bOffset == BITS_PER_BYTE) {
          bOffset = 0;
          ++wOffset;
        }
      }
      return StatusCode::SUCCESS;
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint16_t readoutOrder(uint16_t physical_order) {
      return (4 * ((static_cast<uint16_t>((static_cast<float>(physical_order) / 8.0))) % 4) +
              static_cast<uint16_t>(static_cast<float>(physical_order) / 32.0) + 16 * (physical_order % 8));
    }
  }  // namespace detail

  namespace checks {
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isNonLiteZS(uint8_t mode) { return (mode == 10 || mode == 11); }
  }  // namespace checks

  namespace unpackers {
    ALPAKA_FN_HOST_ACC StatusCode unpackZeroSuppressed2(uint32_t chanIdx,
                                                        const uint8_t* channel_data,
                                                        uint16_t channel_length,
                                                        uint32_t channel_offset,
                                                        SiStripDigiView stripDigis,
                                                        uint32_t* aoff,
                                                        uint16_t stripStart,
                                                        bool isNonLite,
                                                        uint8_t mode,
                                                        uint8_t packetCode = 0) {
      if ((isNonLite && packetCode == PACKET_CODE_ZERO_SUPPRESSED10) ||
          (mode == READOUT_MODE_ZERO_SUPPRESSED_LITE10 || mode == READOUT_MODE_ZERO_SUPPRESSED_LITE10_CMOVERRIDE)) {
        // printf("[%i] unpackZSB<10>\n", *aoff);
        return detail::unpackZSB<10>(
            chanIdx, channel_data, channel_length, channel_offset, stripDigis, aoff, (isNonLite ? 7 : 2), stripStart);
      } else if (mode == READOUT_MODE_PREMIX_RAW) {
        // printf("[%i] unpackZSB<16>\n", *aoff);
        return detail::unpackZSW<16>(
            chanIdx, channel_data, channel_length, channel_offset, stripDigis, aoff, 7, stripStart);
      } else {  // 8bit
        uint8_t bits_shift = 0;
        if (isNonLite) {
          if (packetCode == PACKET_CODE_ZERO_SUPPRESSED8_TOPBOT)
            bits_shift = 1;
          else if (packetCode == PACKET_CODE_ZERO_SUPPRESSED8_BOTBOT)
            bits_shift = 2;
        } else {  // lite
          if (mode == READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT ||
              mode == READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT_CMOVERRIDE)
            bits_shift = 1;
          else if (mode == READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT ||
                   mode == READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT_CMOVERRIDE)
            bits_shift = 2;
        }
        // printf("[%i] unpackZSB<8> [isNonLite %i] [packetCode %i] [mode %i] [channel_length %i] [channel_offset %i]\n", *aoff, isNonLite, packetCode, mode, channel_length, channel_offset);
        auto st = detail::unpackZSW<8>(chanIdx,
                                       channel_data,
                                       channel_length,
                                       channel_offset,
                                       stripDigis,
                                       aoff,
                                       (isNonLite ? 7 : 2),
                                       stripStart,
                                       bits_shift);
        if (isNonLite && packetCode == 0 && StatusCode::SUCCESS == st) {
          // workaround for a pre-2015 bug in the packer: assume default ZS packing
          // printf("[%i] ZERO_PACKET_CODE\n", *aoff);
          return StatusCode::ZERO_PACKET_CODE;
        }
        return st;
      }
    }
  }  // namespace unpackers
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip::fedchannelunpacker

// kernels and related objects
namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace cms::alpakatools;

  constexpr uint16_t invalidStrip = std::numeric_limits<uint16_t>::max();
  constexpr uint16_t invalidFed = std::numeric_limits<uint16_t>::max();
  constexpr uint16_t badBit = (1 << 15);
  constexpr uint16_t stripIndexMask = 0x7FFF;

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint16_t fedIndex(uint16_t fedId) { return fedId - FED_ID_MIN; }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t stripIndex(uint16_t fedID, uint8_t fedCH, uint16_t strip) {
    return fedIndex(fedID) * FEDCH_PER_FED * STRIPS_PER_FEDCH + fedCH * STRIPS_PER_FEDCH + (strip % STRIPS_PER_FEDCH);
  }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t apvIndex(uint16_t fed, uint8_t channel, uint16_t strip) {
    return fedIndex(fed) * APVS_PER_FEDCH * FEDCH_PER_FED + APVS_PER_CHAN * channel +
           (strip % STRIPS_PER_FEDCH) / STRIPS_PER_APV;
  }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t channelIndex(uint16_t fedId, uint8_t fedCh) {
    return fedIndex(fedId) * FEDCH_PER_FED + fedCh;
  }

  class SiStripKer_applyQualConds {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST_ACC void operator()(TAcc const& acc,
                                       uint32_t* blockStripN,
                                       uint32_t* invalidFedCh,
                                       uint8_t* fedChannelsData,
                                       SiStripMappingView mapping,
                                       const DetToFeds* qualityConditions) const {
      //
      for (auto chan : uniform_elements(acc, mapping.metadata().size())) {
        const auto fedId = mapping.fedID(chan);
        const auto fedCh = mapping.fedCh(chan);

        uint32_t index = channelIndex(fedId, fedCh);

        if (qualityConditions->qualityOk[index]) {
          const uint32_t fedChOfs = mapping.fedChOfs(chan);
          const uint32_t fedChDataOfsBuf = mapping.fedChDataOfsBuf(chan);

          bool isNonLite = fedchannelunpacker::checks::isNonLiteZS(mapping.readoutMode(chan));

          FEDChannel fedChan(fedChannelsData + fedChDataOfsBuf,
                             fedChOfs,
                             isNonLite ? FEDChannel::ZSROMode::nonLite : FEDChannel::ZSROMode::lite);

          // Calculate the number of strips in the channel
          uint8_t numBits = 8;
          const uint8_t packetCode = fedChan.packetCode();
          const auto roMode = mapping.readoutMode(chan);
          if ((isNonLite && packetCode == PACKET_CODE_ZERO_SUPPRESSED10) ||
              (roMode == READOUT_MODE_ZERO_SUPPRESSED_LITE10 ||
               roMode == READOUT_MODE_ZERO_SUPPRESSED_LITE10_CMOVERRIDE)) {
            // unpackZSB<10>
            numBits = 10;
          } else if (roMode == READOUT_MODE_PREMIX_RAW) {
            numBits = 16;
          }
          uint16_t chStripNb = fedChan.stripsInCh(numBits);
          mapping.fedChStripsN(chan) = chStripNb;

          // uint8_t chLength = fedChan.length();
          // printf("#chLen,%i,%i,%i,%i,%i\n", fedId, fedCh, fedChOfs, chStripNb, blockStripN[0]);
          // alpaka::atomicAdd(acc, blockStripN, static_cast<uint32_t>(chStripNb), alpaka::hierarchy::Grids{});
        } else {
          // Atomic add the total number of strips which cannot be unpacked (unlikely case)
          mapping.fedID(chan) = invalidFed;
          mapping.fedChStripsN(chan) = 0;
          // printf("#invFed,%i,%i,%i,%i\n", index, fedId, fedCh, invalidFedCh[0]);
          alpaka::atomicAdd(acc, invalidFedCh, 1u, alpaka::hierarchy::Grids{});
        }
      }
    }
  };

  class SiStripKer_init {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST_ACC void operator()(TAcc const& acc,
                                       const float r_channelThreshold,
                                       const float r_seedThreshold,
                                       const float r_clusterThresholdSquared,
                                       const uint8_t r_maxSequentialHoles,
                                       const uint8_t r_maxSequentialBad,
                                       const uint8_t r_maxAdjacentBad,
                                       const float r_minGoodCharge,
                                       const uint32_t r_clusterSizeLimit,
                                       StripClustersAuxView clusterDataObj) const {
      if (once_per_grid(acc)) {
        // Initialize the members of the clusterizer
        clusterDataObj.channelThreshold() = r_channelThreshold;
        clusterDataObj.seedThreshold() = r_seedThreshold;
        clusterDataObj.clusterThresholdSquared() = r_clusterThresholdSquared;
        clusterDataObj.maxSequentialHoles() = r_maxSequentialHoles;
        clusterDataObj.maxSequentialBad() = r_maxSequentialBad;
        clusterDataObj.maxAdjacentBad() = r_maxAdjacentBad;
        clusterDataObj.minGoodCharge() = r_minGoodCharge;
        clusterDataObj.clusterSizeLimit() = r_clusterSizeLimit;
      }
    }
  };

  class SiStripKer_unpackZS2 {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST_ACC void operator()(TAcc const& acc,
                                       uint8_t* fedChannelsData,
                                       SiStripDigiView stripDigis,
                                       SiStripMappingConstView mapping,
                                       const Data* data
                                       //  SiStripClusterizerConditionsData_fedchConstView Data_fedch,
                                       //  SiStripClusterizerConditionsData_stripConstView Data_strip,
                                       //  SiStripClusterizerConditionsData_apvConstView Data_apv
    ) const {
      // Loop over the FEDChannel collection to be digitized
      for (auto chan : uniform_elements(acc, mapping.metadata().size())) {
        // for (uint32_t chan=0; chan<(uint32_t)mapping.metadata().size(); chan++) {
        const auto fedId = mapping.fedID(chan);
        const auto fedCh = mapping.fedCh(chan);
        // const auto detId = mapping.detID(chan);

        // reject strips which are in the conditions but not in the fed data collection
        if (fedId == invalidFed) {
          continue;
        }

        const auto ipair = data->iPair[channelIndex(fedId, fedCh)];
        int ipoff = STRIPS_PER_FEDCH * ipair;

        const uint32_t fedChDataOfsBuf = mapping.fedChDataOfsBuf(chan);
        uint16_t fedChOfs = mapping.fedChOfs(chan);
        const uint8_t mode = mapping.readoutMode(chan);
        const bool isNonLite = fedchannelunpacker::checks::isNonLiteZS(mode);

        const FEDChannel fedChan(fedChannelsData + fedChDataOfsBuf, fedChOfs, FEDChannel::ZSROMode(isNonLite));
        const uint8_t packetCode = fedChan.packetCode();

        uint32_t absoluteOffset = 0;
        if (chan > 0) [[likely]] {
          absoluteOffset = mapping.fedChStripsN(chan - 1);
        }

        auto retCode = fedchannelunpacker::unpackers::unpackZeroSuppressed2(chan,
                                                                            fedChan.data(),
                                                                            fedChan.length(),
                                                                            fedChan.offset(),
                                                                            stripDigis,
                                                                            &absoluteOffset,
                                                                            ipoff,
                                                                            isNonLite,
                                                                            mode,
                                                                            packetCode);
        if (retCode != fedchannelunpacker::StatusCode::SUCCESS) {
          printf("[%i] [fedID %hu] [fedCH %hhu] - Returned %i \n", chan, fedId, fedCh, (int)retCode);
        }
      }  // data != nullptr && len > 0
    }
  };

  class SiStripKer_setSeedStrips {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  SiStripDigiConstView stripDigi,
                                  StripClustersAuxView clusterDataObj,
                                  SiStripMappingConstView mapping,
                                  const Data* data) const {
      auto nStrips = stripDigi.metadata().size();
      const float seedThreshold = clusterDataObj.seedThreshold();
      for (auto i : uniform_elements(acc, nStrips)) {
        clusterDataObj.seedStripsMask(i) = 0;
        clusterDataObj.seedStripsNCMask(i) = 0;
        clusterDataObj.prefixSeedStripsNCMask(i) = 0;

        const auto chan_ = stripDigi.channel(i);
        const auto fedId = mapping.fedID(chan_);
        const auto fedCh = mapping.fedCh(chan_);
        const auto stripID = stripDigi.stripId(i);

        const uint32_t idx = stripIndex(fedId, fedCh, stripID);
        const uint16_t noise_tmp = data->noise[idx];
        const bool isBad = (noise_tmp & badBit) > 0;

        if (!isBad) {
          const float noise_i = 0.1f * (noise_tmp & ~badBit);
          const uint8_t adc_i = stripDigi.adc(i);
          clusterDataObj.seedStripsMask(i) = (adc_i >= static_cast<uint8_t>(noise_i * seedThreshold)) ? 1 : 0;
          clusterDataObj.seedStripsNCMask(i) = clusterDataObj.seedStripsMask(i);
        }
      }
    }
  };

  class SiStripKer_setNCSeedStrips {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  SiStripDigiConstView stripDataObj,
                                  StripClustersAuxView clusterDataObj,
                                  SiStripMappingConstView mapping) const {
      auto channels = stripDataObj.channel();
      // Loop over the strips
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

  class SiStripKer_setNCStripIndex {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, StripClustersAuxView clusterDataObj) const {
      // Loop over the strips
      for (auto stripIdx : uniform_elements(acc, clusterDataObj.metadata().size())) {
        if (clusterDataObj.seedStripsNCMask(stripIdx) == 1) {
          const int index = (clusterDataObj.prefixSeedStripsNCMask(stripIdx) - 1);
          clusterDataObj.seedStripsNCIndex(index) = stripIdx;
        }
      }
    }
  };

  class SiStripKer_makeCandidates {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  const uint32_t kMaxSeedStrips,
                                  SiStripDigiConstView stripDataObj,
                                  StripClustersAuxConstView clusterDataObj,
                                  SiStripClusterView clusters,
                                  SiStripMappingConstView mapping,
                                  const Data* conditionsData) const {
      //
      const int32_t nStrips = stripDataObj.metadata().size();
      const uint32_t nSeedStripsNC = (kMaxSeedStrips < clusterDataObj.prefixSeedStripsNCMask(nStrips - 1))
                                         ? kMaxSeedStrips
                                         : clusterDataObj.prefixSeedStripsNCMask(nStrips - 1);

      const auto& chanArr = stripDataObj.channel();
      const auto& stripIdArr = stripDataObj.stripId();
      const auto& adcArr = stripDataObj.adc();

      const float channelThreshold = clusterDataObj.channelThreshold();
      const float clusterThresholdSquared = clusterDataObj.clusterThresholdSquared();
      const uint8_t maxSequentialHoles = clusterDataObj.maxSequentialHoles();
      const uint16_t clusterSizeLimit = clusterDataObj.clusterSizeLimit();

      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        clusters.nClusterCandidates() = nSeedStripsNC;
        clusters.maxClusterSize() = clusterSizeLimit;
      }

      // Loop over only the non-contiguous strips (flagged in setStripIndex)
      for (auto i : uniform_elements(acc, nSeedStripsNC)) {
        const uint32_t chan_ = clusterDataObj.seedStripsNCIndex(i);

        const auto chan = chanArr[chan_];
        const auto fedId = mapping.fedID(chan);
        const auto fedCh = mapping.fedCh(chan);
        const auto detId = mapping.detID(chan);
        const auto stripId = stripIdArr[chan_];
        //
        const uint32_t idx = stripIndex(fedId, fedCh, stripId);
        const uint16_t noise_tmp = conditionsData->noise[idx];
        const float noise_i = 0.1f * (noise_tmp & ~badBit);

        // Calculate the accumulated noise2 and ADC
        uint16_t clSize = 1;
        float noiseSquared_i = noise_i * noise_i;
        float adcSum_i = adcArr[chan_];
        int32_t testIndex = chan_ - 1;

        auto addtocluster = [&](int& indexLR) {
          const auto test_chan = chanArr[testIndex];
          const auto test_fedId = mapping.fedID(test_chan);
          const auto test_fedCh = mapping.fedCh(test_chan);
          const auto test_stripId = stripIdArr[testIndex];

          const uint32_t test_idx = stripIndex(test_fedId, test_fedCh, test_stripId);
          const uint16_t test_noise_tmp = conditionsData->noise[test_idx];
          const bool test_isBad = (test_noise_tmp & badBit) > 0;

          const float test_noise_i = 0.1f * (test_noise_tmp & ~badBit);
          const uint8_t testADC = adcArr[testIndex];

          if (!test_isBad && (testADC >= static_cast<uint8_t>(test_noise_i * channelThreshold))) {
            ++clSize;
            indexLR = testIndex;
            noiseSquared_i += test_noise_i * test_noise_i;
            adcSum_i += testADC;
          }
        };

        // find left boundary
        int32_t indexLeft = chan_;

        // if (stripIdArr[testIndex] == invalidStrip && testIndex >= 0) {
        //   testIndex -= 2;
        // }

        if (testIndex >= 0) {
          const auto testchan = chanArr[testIndex];
          const auto testDetId = mapping.detID(testchan);

          int16_t rangeLeft = stripIdArr[indexLeft] - stripIdArr[testIndex] - 1;
          bool isSameDetLeft = (detId == testDetId);

          while (isSameDetLeft && (rangeLeft >= 0) && (rangeLeft <= maxSequentialHoles) &&
                 (clSize < (clusterSizeLimit + 1))) {
            addtocluster(indexLeft);
            --testIndex;
            // if (testIndex >= 0 && stripIdArr[testIndex] == invalidStrip) {
            //   testIndex -= 2;
            // }
            if (testIndex >= 0) {
              rangeLeft = stripIdArr[indexLeft] - stripIdArr[testIndex] - 1;
              const auto newchan = chanArr[testIndex];
              const auto newdet = mapping.detID(newchan);
              isSameDetLeft = (detId == newdet);
            } else {
              isSameDetLeft = false;
            }
          }  // while loop
        }  // testIndex >= 0

        // find right boundary
        int32_t indexRight = chan_;
        testIndex = chan_ + 1;

        // if (stripIdArr[testIndex] == invalidStrip && testIndex < nStrips) {
        //   testIndex += 2;
        // }

        if (testIndex < nStrips) {
          const auto testchan = chanArr[testIndex];
          const auto testDetId = mapping.detID(testchan);

          int16_t rangeRight = stripIdArr[testIndex] - stripIdArr[indexRight] - 1;
          bool isSameDetRight = (detId == testDetId);

          while (isSameDetRight && (rangeRight >= 0) && (rangeRight <= maxSequentialHoles) &&
                 (clSize < (clusterSizeLimit + 1))) {
            addtocluster(indexRight);
            ++testIndex;
            // if (testIndex < nStrips && stripIdArr[testIndex] == invalidStrip) {
            //   testIndex += 2;
            // }
            if (testIndex < nStrips) {
              rangeRight = stripIdArr[testIndex] - stripIdArr[indexRight] - 1;
              const auto newchan = chanArr[testIndex];
              const auto newdet = mapping.detID(newchan);
              isSameDetRight = (detId == newdet);
            } else {
              isSameDetRight = false;
            }
          }  // while loop
        }  // testIndex < nStrips

        clusters.clusterIndex(i) = indexLeft;
        clusters.clusterSize(i) = indexRight - indexLeft + 1;
        clusters.clusterDetId(i) = detId;
        clusters.firstStrip(i) = stripIdArr[indexLeft];
        // Flag candidates which do not pass (cluster noise threshold) && (cluster size) conditions. Max number of holes already accounted in candidate finder
        // Floating point approximation leads sensitivity on the above condition for O(12/500) events, with NCluster deviation w.r.t. legacy module of O(1).
        clusters.candidateAccepted(i) = (noiseSquared_i * clusterThresholdSquared <= adcSum_i * adcSum_i) &&
                                        (clusters.clusterSize(i) <= clusterSizeLimit);
        clusters.candidateAcceptedPrefix(i) = static_cast<uint32_t>(clusters.candidateAccepted(i));
      }  // i < nSeedStripsNC
    }
  };

  class SiStripKer_endCandidates {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  SiStripDigiView stripDataObj,
                                  StripClustersAuxConstView clusterDataObj,
                                  SiStripClusterView clusters,
                                  SiStripMappingConstView mapping,
                                  const Data* conditionsData) const {
      //
      constexpr uint8_t adc_low_saturation = 254;
      constexpr uint8_t adc_high_saturation = 255;
      constexpr int charge_low_saturation = 253;
      constexpr int charge_high_saturation = 1022;
      //
      const auto& clusterIndexArr = clusters.clusterIndex();

      for (auto i : uniform_elements(acc, clusters.nClusterCandidates())) {
        if (clusters.candidateAccepted(i)) {
          const uint32_t indexLeft = clusterIndexArr[i];
          const uint16_t size = clusters.clusterSize(i);

          if (i > 0 && clusterIndexArr[i - 1] == indexLeft) {
            clusters.candidateAccepted(i) = false;  // ignore duplicates
            clusters.candidateAcceptedPrefix(i) = 0;
          } else {
            float chargeSum = 0.0f;
            uint16_t sumx = 0;
            uint16_t suma = 0;

            uint16_t j = 0;
            for (uint16_t k = 0; k < size; k++) {
              uint32_t index = indexLeft + k;
              const auto chan = stripDataObj.channel(index);
              const auto fedId = mapping.fedID(chan);
              const auto fedCh = mapping.fedCh(chan);
              const auto stripId = stripDataObj.stripId(index);

              // ThreeThresholdAlgorithm::applyGains
              const float gain_j = conditionsData->gain[apvIndex(fedId, fedCh, stripId)];
              uint8_t amplitudes_j = stripDataObj.adc(index);
              const uint16_t charge = static_cast<uint16_t>(static_cast<float>(amplitudes_j) / gain_j + 0.5f);

              if (amplitudes_j < adc_low_saturation) {
                amplitudes_j = ((charge > charge_high_saturation)
                                    ? adc_high_saturation
                                    : (charge > charge_low_saturation ? adc_low_saturation : charge));
              }

              // Overrides the ADC value in the StripDigi with the corrected amplitude
              stripDataObj.adc(index) = amplitudes_j;
              // clusters.clusterADCs(i)[j] = amplitudes_j;

              // SiStripCluster::initQB()
              chargeSum += static_cast<float>(amplitudes_j);
              sumx += j * amplitudes_j;
              suma += amplitudes_j;
              j++;
            }

            clusters.charge(i) = chargeSum;

            const auto chanL = stripDataObj.channel(indexLeft);
            const auto fedIdL = mapping.fedID(chanL);
            const auto fedChL = mapping.fedCh(chanL);

            // Flags for siStripClusterTools::chargePerCM in current candidate
            clusters.candidateAccepted(i) = (clusterDataObj.minGoodCharge() <= 0 ||
                                             ((chargeSum * conditionsData->invthick[channelIndex(fedIdL, fedChL)]) >
                                              clusterDataObj.minGoodCharge()));

            // SiStripCluster::initQB() -> barycenter_
            const float bary_i = static_cast<float>(sumx) / static_cast<float>(suma);
            clusters.barycenter(i) =
                static_cast<float>(stripDataObj.stripId(indexLeft) & stripIndexMask) + bary_i + 0.5f;

            clusters.clusterSize(i) = j;
          }  // not a duplicate cluster
        }
      }  // i < nSeedStripsNC
    }
  };

  class SiStripKer_blkPfxScan {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, StripClustersAuxView clusterDataObj) const {
      //
      // This kernel must run with a single block
      [[maybe_unused]] const uint32_t blockIdxLocal(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      ALPAKA_ASSERT_ACC(0 == blockIdxLocal);
      [[maybe_unused]] const uint32_t gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      ALPAKA_ASSERT_ACC(1 == gridDimension);
      // auto thIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];

      // For the prefix scan algorithm
      constexpr int warpSize = cms::alpakatools::warpSize;
      constexpr int blockSize = warpSize * warpSize;  // assume 32*32 = 1024

      // For Phase1 there are 1856 pixel modules
      // For Phase2 there are up to 4000 pixel modules
      const int numberOfModules = clusterDataObj.metadata().size();
      const int prefixScanUpperLimit = ((numberOfModules / blockSize) + 1) * blockSize;
      ALPAKA_ASSERT_ACC(numberOfModules < prefixScanUpperLimit);

      // Use N single-block prefix scan, then update all blocks after the first one.
      auto& ws = alpaka::declareSharedVar<uint32_t[warpSize], __COUNTER__>(acc);
      auto clusModuleStart = clusterDataObj.seedStripsNCMask();
      auto prefix = clusterDataObj.prefixSeedStripsNCMask();
      int leftModules = numberOfModules;
      // First pass
      uint32_t offset = 0;
      while (leftModules > blockSize) {
        // if (thIdx == 0){
        //   printf("[%i] | numberOfModules %i | leftModules %i\n", thIdx, numberOfModules, leftModules);
        // }
        auto clusModuleStart_chunk = clusModuleStart.subspan(offset, blockSize);
        auto prefix_chunk = prefix.subspan(offset, blockSize);
        cms::alpakatools::blockPrefixScan(acc, clusModuleStart_chunk.data(), prefix_chunk.data(), blockSize, ws);
        leftModules -= blockSize;
      }
      cms::alpakatools::blockPrefixScan(acc, clusModuleStart, prefix, leftModules, ws);

      // Second pass
      // The first blockSize modules are properly accounted by the blockPrefixScan.
      // The additional modules need to be corrected adding the cuulative value from the last module of the previous block.
      for (int doneModules = blockSize; doneModules < numberOfModules; doneModules += blockSize) {
        int first = doneModules;
        int last = (doneModules + blockSize) < numberOfModules ? (doneModules + blockSize) : numberOfModules;
        for (int i : cms::alpakatools::independent_group_elements(acc, first, last)) {
          clusterDataObj.prefixSeedStripsNCMask(i) += clusterDataObj.prefixSeedStripsNCMask(first - 1);
          // printf("[%i] - prefixSeedStripsNCMask(%i) = %i\n", thIdx, i, clusterDataObj.prefixSeedStripsNCMask(i));
        }
        alpaka::syncBlockThreads(acc);
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

// kernels launchers
namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace ::sistrip;
  using namespace cms::alpakatools;

  SiStripRawToClusterAlgo::SiStripRawToClusterAlgo(const edm::ParameterSet& clustPar)
      : channelThreshold_(clustPar.getParameter<double>("ChannelThreshold")),
        seedThreshold_(clustPar.getParameter<double>("SeedThreshold")),
        clusterThresholdSquared_(std::pow(clustPar.getParameter<double>("ClusterThreshold"), 2.0f)),
        maxSequentialHoles_(clustPar.getParameter<unsigned>("MaxSequentialHoles")),
        maxSequentialBad_(clustPar.getParameter<unsigned>("MaxSequentialBad")),
        maxAdjacentBad_(clustPar.getParameter<unsigned>("MaxAdjacentBad")),
        maxClusterSize_(clustPar.getParameter<unsigned>("MaxClusterSize")),
        minGoodCharge_(clusterChargeCut(clustPar)),
        kMaxSeedStrips_(clustPar.getParameter<uint32_t>("MaxSeedStrips")) {
    // Checks non-sensical parameters
    if (maxClusterSize_ > 768) {
      throw cms::Exception("SiStripRawToClstAlg", "MaxClusterSize must be <= 768");
    }
  }

  void SiStripRawToClusterAlgo::prepareUnpackCluster(Queue& queue,
                                                     const DetToFeds* conditions_DetToFeds,
                                                     std::unique_ptr<PortableFEDMover> rFEDChMover) {
    // Move ownership of the host-data container this class
    fedChMover_ = std::move(rFEDChMover);

    // Move the data to the device
    fedBuffer_d_.emplace(cms::alpakatools::make_device_buffer<uint8_t[]>(queue, fedChMover_->bufferSize()));
    alpaka::memcpy(queue, *fedBuffer_d_, fedChMover_->buffer(), fedChMover_->bufferSize());

    // Move fedchannels to the device
    stripMapping_d_ = cms::alpakatools::moveToDeviceAsync(queue, fedChMover_->mapping());

    // Apply quality conditions to mapping and calculate the number of strips to unpack
    uint32_t divider = 32u;
    uint32_t groups = divide_up_by(fedChMover_->channelNb(), divider);
    auto workDiv = make_workdiv<Acc1D>(groups, divider);

    auto nstrips_d = make_device_buffer<uint32_t>(queue);
    alpaka::memset(queue, nstrips_d, 0);

    auto invalidFedChN_d = make_device_buffer<uint32_t>(queue);
    alpaka::memset(queue, invalidFedChN_d, 0);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripKer_applyQualConds{},
                        nstrips_d.data(),
                        invalidFedChN_d.data(),
                        fedBuffer_d_->data(),
                        stripMapping_d_->view(),
                        conditions_DetToFeds);

    const uint32_t threads = 1024u;
    const uint32_t nBlocks = divide_up_by(stripMapping_d_->view().metadata().size(), threads);
    const auto workDivMultiBlock = make_workdiv<Acc1D>(nBlocks, threads);
    auto blockCounter_d = make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, blockCounter_d, 0);
    alpaka::exec<Acc1D>(queue,
                        workDivMultiBlock,
                        multiBlockPrefixScan<uint32_t>(),
                        stripMapping_d_->const_view().fedChStripsN().data(),
                        stripMapping_d_->view().fedChStripsN().data(),
                        (uint32_t)stripMapping_d_->view().metadata().size(),
                        (int32_t)nBlocks,
                        blockCounter_d.data(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));

    // To do: merge the two kernels for applyQualConds and sum into a single one.

    // Get the number of strips to unpack
    nStrips_h_ = cms::alpakatools::make_host_buffer<uint32_t>(queue);
    const uint32_t size = stripMapping_d_->view().metadata().size();
    auto viewSrc = make_device_view(queue, stripMapping_d_->view().fedChStripsN(size - 1));
    auto viewDst = make_host_view(*nStrips_h_->data());
    alpaka::memcpy(queue, viewDst, viewSrc);
  }

  void SiStripRawToClusterAlgo::unpackStrips(Queue& queue, const Data* conditions_Data) {
    // std::cout << "#kerStrips," << nStrips_h->data()[0] << std::endl;
    // std::cout << "#invalidFed," << invalidFedChN_d.data()[0] << std::endl;

    // Allocate the SiStripDigi collection on device
    const uint32_t nStrips = nStrips_h_->data()[0];
    digis_d_ = std::make_unique<SiStripDigiDevice>(nStrips, queue);

    // Run the unpacking kernel
    uint32_t divider = 32u;
    uint32_t nChannels = (*stripMapping_d_)->metadata().size();
    uint32_t groups = divide_up_by(nChannels, divider);
    auto workDiv = make_workdiv<Acc1D>(groups, divider);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripKer_unpackZS2{},
                        fedBuffer_d_->data(),
                        digis_d_->view(),
                        stripMapping_d_->const_view(),
                        conditions_Data);

    // dumpUnpackedStrips(queue, digis_d_.get()); // (for debugging)

    // Allocate and initialize the StripClustersAux collection
    sClustersAux_d_ = std::make_unique<StripClustersAuxDevice>(nStrips, queue);
    // LogDebug("sClustersAux") << "Size of StripClustersAuxDevice (bytes): " << alpaka::getExtentProduct(sClustersAux_d_->buffer()) * sizeof(std::byte);
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(1u, 1u),
                        SiStripKer_init{},
                        channelThreshold_,
                        seedThreshold_,
                        clusterThresholdSquared_,
                        maxSequentialHoles_,
                        maxSequentialBad_,
                        maxAdjacentBad_,
                        minGoodCharge_,
                        maxClusterSize_,
                        sClustersAux_d_->view());

    // Cluster seeding
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripKer_setSeedStrips{},
                        digis_d_->const_view(),
                        sClustersAux_d_->view(),
                        stripMapping_d_->const_view(),
                        conditions_Data);

    // Un-seed any contiguous strips in the same detector
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripKer_setNCSeedStrips{},
                        digis_d_->const_view(),
                        sClustersAux_d_->view(),
                        stripMapping_d_->const_view());

    // Calculate the discrete integral (prefix sum) of seedStripsNCMask.
    // When the integral increase AND I am at a non-contigous strip, the beginning of new cluster is marked.
    const uint32_t nThreads = 1024u;
    const int32_t nBlocks = divide_up_by(nStrips, nThreads);
    auto blockCounter_d = make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, blockCounter_d, 0);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(make_workdiv<Acc1D>(nBlocks, nThreads),
                                                    multiBlockPrefixScan<uint32_t>(),
                                                    sClustersAux_d_->const_view().seedStripsNCMask().data(),
                                                    sClustersAux_d_->view().prefixSeedStripsNCMask().data(),
                                                    nStrips,
                                                    nBlocks,
                                                    blockCounter_d.data(),
                                                    alpaka::getPreferredWarpSize(alpaka::getDev(queue))));

    // Get the total number of non-contiguous seeds (ready on the host in produce)
    nSeeds_h_ = cms::alpakatools::make_host_buffer<uint32_t>(queue);
    auto viewSrc = make_device_view(queue, sClustersAux_d_->view().prefixSeedStripsNCMask(nStrips - 1));
    //alpaka::ViewPlainPtr<alpaka::DevCpu, unsigned int, std::integral_constant<unsigned long, 0>, unsigned int>
    auto viewDst = make_host_view(*nSeeds_h_->data());
    //alpaka::ViewPlainPtr<alpaka::DevCpu, unsigned int, std::integral_constant<unsigned long, 0>, unsigned int>
    alpaka::memcpy(queue, viewDst, viewSrc);

    // Find index of the non-contiguous strip seeds
    alpaka::exec<Acc1D>(queue, workDiv, SiStripKer_setNCStripIndex{}, sClustersAux_d_->view());

    // dumpSeeds(queue, digis_d_.get(), sClustersAux_d_.get()); // (for debugging)
  }

  std::unique_ptr<SiStripClusterDevice> SiStripRawToClusterAlgo::makeClusters(Queue& queue,
                                                                              const Data* conditions_Data) {
    // The maximum number of clusters is set to kMaxSeedStrips
    auto clusters_d = std::make_unique<SiStripClusterDevice>(kMaxSeedStrips_, queue);
    clusters_d->zeroInitialise(queue);

    // The number of seed over which to loop for clusters is the min between the number of strips and the kMaxSeeds
    const uint32_t nStrips = nStrips_h_->data()[0];
    const uint32_t nSeeds = std::min(kMaxSeedStrips_, nStrips);

    uint32_t divider = 256u;
    uint32_t groups = divide_up_by(nSeeds, divider);
    auto workDiv = make_workdiv<Acc1D>(groups, divider);

    // Three-threshold clusterization algo
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripKer_makeCandidates{},
                        kMaxSeedStrips_,
                        digis_d_->const_view(),
                        sClustersAux_d_->const_view(),
                        clusters_d->view(),
                        stripMapping_d_->const_view(),
                        conditions_Data);

    // dumpClusters(queue, clusters_d.get(), digis_d_.get());

    // Apply the conditions
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripKer_endCandidates{},
                        digis_d_->view(),
                        sClustersAux_d_->const_view(),
                        clusters_d->view(),
                        stripMapping_d_->const_view(),
                        conditions_Data);

    // Fill the prefix indexes for the candidateAccepted
    const uint32_t nThreads = 1024u;
    const int32_t nBlocks = divide_up_by(kMaxSeedStrips_, nThreads);
    auto blockCounter_d = make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, blockCounter_d, 0);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(make_workdiv<Acc1D>(nBlocks, nThreads),
                                                    multiBlockPrefixScan<uint32_t>(),
                                                    clusters_d->const_view().candidateAcceptedPrefix().data(),
                                                    clusters_d->view().candidateAcceptedPrefix().data(),
                                                    kMaxSeedStrips_,
                                                    nBlocks,
                                                    blockCounter_d.data(),
                                                    alpaka::getPreferredWarpSize(alpaka::getDev(queue))));

    // Store the total number of good cluster candidates into the scalar of the StripDigi SoA
    auto viewSrc_realClusters =
        make_device_view(queue, clusters_d->view().candidateAcceptedPrefix(kMaxSeedStrips_ - 1));
    auto viewDst_realClusters = make_device_view(queue, digis_d_->view().nbGoodCandidates());
    alpaka::memcpy(queue, viewDst_realClusters, viewSrc_realClusters);

    // Store also the number of cluster candidates, in order to reduce as much as possible the loop for the slimming in the legacy converter
    auto viewSrc_candidatesN = make_device_view(queue, clusters_d->view().nClusterCandidates());
    auto viewDst_candidatesN = make_device_view(queue, digis_d_->view().nbCandidates());
    alpaka::memcpy(queue, viewDst_candidatesN, viewSrc_candidatesN);

    // dumpClusters(queue, clusters_d.get(), digis_d_.get());
    return clusters_d;
  }

  std::unique_ptr<SiStripDigiDevice> SiStripRawToClusterAlgo::releaseDigiAmplitudes(Queue& queue) {
    return std::move(digis_d_);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

// Debugging functions
namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  void SiStripRawToClusterAlgo::dumpUnpackedStrips(Queue& queue, SiStripDigiDevice* digis_d) {
    const int digisSize = digis_d->const_view().metadata().size();
    auto digis_h = SiStripDigiHost(digisSize, queue);
    alpaka::memcpy(queue, digis_h.buffer(), digis_d->const_buffer());
    alpaka::wait(queue);
    std::ostringstream dumpMsg("");
    dumpMsg << "[SiStripRawToClusterAlgo::unpackStrips] Dumping unpacked strips\n";
    dumpMsg << "Allocated " << digisSize << " strips\n";
    dumpMsg << "i,chan,stripId,adc\n";
    for (int i = 0; i < digisSize; ++i) {
      if (i < std::min(digisSize, 100) || i > (digisSize - 100)) {
        dumpMsg << i << "," << (int)(digis_h->channel(i)) << "," << (int)(digis_h->stripId(i)) << ","
                << (int)digis_h->adc(i) << "\n";
      }
    }
    std::cout << dumpMsg.str();
  }

  void SiStripRawToClusterAlgo::dumpSeeds(Queue& queue,
                                          SiStripDigiDevice* digis_d,
                                          StripClustersAuxDevice* sClustersAux_d) {
    // Store the size of the digi to avoid repetitions
    const int digisSize = digis_d->const_view().metadata().size();
    auto digis_h = SiStripDigiHost(digisSize, queue);
    alpaka::memcpy(queue, digis_h.buffer(), digis_d->const_buffer());
    // Seed table and digis have the same size
    auto sClustersAux_h = StripClustersAuxHost(digisSize, queue);
    alpaka::memcpy(queue, sClustersAux_h.buffer(), sClustersAux_d->const_buffer());
    alpaka::wait(queue);

    std::ostringstream dumpMsg("");
    dumpMsg << "[SiStripRawToClusterAlgo::setSeedsAndMakeIndexes] Dumping seeds table\n";
    dumpMsg << "i\tadc\tchan\tstripId\tseed\tseedNC\tpfxNCMask\tseedNCIndex\n";
    for (int i = 0; i < digisSize; ++i) {
      if (i < 50 || i > (digisSize - 50) || i % 10000 == 0) {
        if (digis_h->stripId(i) != invalidStrip) {
          dumpMsg << i << "\t" << (int)(digis_h->adc(i)) << "\t" << digis_h->channel(i) << "\t\t\t\t"
                  << digis_h->stripId(i) << "\t\t\t" << sClustersAux_h->seedStripsMask(i) << "\t\t\t"
                  << sClustersAux_h->seedStripsNCMask(i) << "\t\t\t\t" << sClustersAux_h->prefixSeedStripsNCMask(i)
                  << "\t\t\t\t\t" << sClustersAux_h->seedStripsNCIndex(i) << "\n";
        }
      }
    }
    std::cout << dumpMsg.str();
  }

  void SiStripRawToClusterAlgo::dumpClusters(Queue& queue,
                                             SiStripClusterDevice* clusters_d,
                                             SiStripDigiDevice* digis_d) {
    // Store the size of the digi to avoid repetitions
    const int clustersPrealloc = clusters_d->view().metadata().size();
    auto clusters_h = SiStripClusterHost(clustersPrealloc, queue);
    alpaka::memcpy(queue, clusters_h.buffer(), clusters_d->const_buffer());

    const uint32_t nStrips = digis_d->view().metadata().size();
    auto digis_h = SiStripDigiHost(nStrips, queue);
    alpaka::memcpy(queue, digis_h.buffer(), digis_d->const_buffer());

    alpaka::wait(queue);

    const int clustersN = clusters_h->nClusterCandidates();

    std::ostringstream dumpMsg("");
    dumpMsg << "#clDump,Pre-allocated:" << clustersPrealloc << ",Candidates:" << clustersN << "\n";
    dumpMsg << "i,cIdx,cSz,cDetId,chg,1st,tCl,tClIdx,bary,|clusterADCs|\n";

    for (int i = 0; i < clustersN; ++i) {
      if (i < 100 || i > (clustersN - 100)) {
        dumpMsg << i << "," << clusters_h->clusterIndex(i) << "," << clusters_h->clusterSize(i) << ","
                << clusters_h->clusterDetId(i) << "," << clusters_h->charge(i) << "," << clusters_h->firstStrip(i)
                << "," << clusters_h->candidateAccepted(i) << "," << clusters_h->candidateAcceptedPrefix(i) << ","
                << clusters_h->barycenter(i) << ",|";
        if (clusters_h->candidateAccepted(i)) {
          for (int j = 0; j < clusters_h->clusterSize(i); ++j) {
            uint32_t index = clusters_h->clusterIndex(i) + j;
            dumpMsg << (int)(digis_h->adc(index));
            if (j != (clusters_h->clusterSize(i) - 1)) {
              dumpMsg << "/";
            }
          }
        } else {
          dumpMsg << "-";
        }
        dumpMsg << "|\n";
      }
    }
    dumpMsg << "#zClDump\n";
    dumpMsg << "#goodCandidates," << digis_h.view().nbGoodCandidates() << ":"
            << clusters_h->candidateAcceptedPrefix(clustersPrealloc - 1) << "\n";

    dumpMsg << "#last\n";
    dumpMsg << "i,cIdx,cSz,cDetId,chg,1st,tCl,tClIdx,bary,|clusterADCs|\n";
    for (int i = clustersPrealloc - 100; i > 0 && i < clustersPrealloc; ++i) {
      dumpMsg << i << "," << clusters_h->clusterIndex(i) << "," << clusters_h->clusterSize(i) << ","
              << clusters_h->clusterDetId(i) << "," << clusters_h->charge(i) << "," << clusters_h->firstStrip(i) << ","
              << clusters_h->candidateAccepted(i) << "," << clusters_h->candidateAcceptedPrefix(i) << ","
              << clusters_h->barycenter(i) << ",|";
      if (clusters_h->candidateAccepted(i)) {
        for (int j = 0; j < clusters_h->clusterSize(i); ++j) {
          uint32_t index = clusters_h->clusterIndex(i) + j;
          dumpMsg << (int)(digis_h->adc(index));
          if (j != (clusters_h->clusterSize(i) - 1)) {
            dumpMsg << "/";
          }
        }
      } else {
        dumpMsg << "-";
      }
      dumpMsg << "|\n";
    }
    std::cout << dumpMsg.str();
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip
