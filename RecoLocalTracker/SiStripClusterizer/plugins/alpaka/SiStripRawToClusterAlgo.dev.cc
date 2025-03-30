#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"
// #include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"

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

    // Unpack Raw with ADCs in whole 8-bit words (8bit and 10-in-16bit)
    template <uint8_t num_bits, typename OUT>
    ALPAKA_FN_HOST_ACC StatusCode unpackRawW(const FEDChannel& channel, OUT&& out, uint8_t bits_shift = 0) {
      constexpr auto num_words = num_bits / 8;
      static_assert(((num_bits % 8) == 0) && (num_words > 0) && (num_words < 3));
      if ((num_words > 1) && ((channel.length() - 3) % num_words)) {
        // LogDebug("FEDBuffer") << "Channel length is invalid. Raw channels have 3 header bytes and " << num_words
        //                       << " bytes per sample. "
        //                       << "Channel length is " << uint16_t(channel.length()) << ".";
        return StatusCode::BAD_CHANNEL_LENGTH;
      }
      const uint8_t* const data = channel.data();
      const uint_fast16_t end = channel.offset() + channel.length();
      for (uint_fast16_t offset = channel.offset() + 3; offset != end; offset += num_words) {
        *out++ = SiStripRawDigi(getADC_W<num_words>(data, offset, bits_shift));
      }
      return StatusCode::SUCCESS;
    }

    // Unpack Raw with ADCs in whole 8-bit words (8bit and 10-in-16bit)
    template <uint8_t num_bits>
    ALPAKA_FN_HOST_ACC StatusCode unpackRawW(const uint8_t* channel_data,
                                             uint16_t channel_length,
                                             uint32_t channel_offset,
                                             StripDigiView out,
                                             int* idx,
                                             uint8_t bits_shift = 0) {
      constexpr auto num_words = num_bits / 8;
      static_assert(((num_bits % 8) == 0) && (num_words > 0) && (num_words < 3));
      if ((num_words > 1) && ((channel_length - 3) % num_words)) {
        // LogDebug("FEDBuffer") << "Channel length is invalid. Raw channels have 3 header bytes and " << num_words
        //                       << " bytes per sample. "
        //                       << "Channel length is " << uint16_t(channel.length()) << ".";
        return StatusCode::BAD_CHANNEL_LENGTH;
      }
      const uint8_t* const data = channel_data;
      const uint_fast16_t end = channel_offset + channel_length;
      for (uint_fast16_t offset = channel_offset + 3; offset != end; offset += num_words) {
        out.adc((*idx)++) = getADC_W<num_words>(data, offset, bits_shift);
      }
      return StatusCode::SUCCESS;
    }

    // Generic implementation for non-whole words (10bit, essentially)
    template <uint_fast8_t num_bits>
    ALPAKA_FN_HOST_ACC StatusCode unpackRawB(int chan,
                                             const uint8_t* channel_data,
                                             uint16_t channel_length,
                                             uint32_t channel_offset,
                                             StripDigiView out,
                                             int* idx) {
      static_assert(num_bits <= 16, "Word length must be between 0 and 16.");
      if (channel_length & 0xF000) {
        // LogDebug("FEDBuffer") << "Channel length is invalid. Channel length is " << uint16_t(channel.length()) << ".";
        return StatusCode::BAD_CHANNEL_LENGTH;
      }
      constexpr uint16_t mask = (1 << num_bits) - 1;
      const uint8_t* const data = channel_data;
      const uint_fast16_t chEnd = channel_offset + channel_length;
      uint_fast16_t wOffset = channel_offset + 3;
      uint_fast8_t bOffset = 0;
      while (((wOffset + 1) < chEnd) || ((chEnd - wOffset) * BITS_PER_BYTE - bOffset >= num_bits)) {
        bOffset += num_bits;
        if ((num_bits > BITS_PER_BYTE) || (bOffset > BITS_PER_BYTE)) {
          bOffset -= BITS_PER_BYTE;
          out.adc(*idx) = getADC_B2<mask>(data, wOffset, bOffset);
          ++wOffset;
        } else {
          out.adc(*idx) = getADC_B1<mask>(data, wOffset, bOffset);
        }
        out.channel(*idx) = chan;
        (*idx)++;
        if (bOffset == BITS_PER_BYTE) {
          bOffset = 0;
          ++wOffset;
        }
      }
      return StatusCode::SUCCESS;
    }

    template <uint8_t num_bits>
    ALPAKA_FN_HOST_ACC StatusCode unpackZSW(int chan,
                                            const uint8_t* channel_data,
                                            uint16_t channel_length,
                                            uint32_t channel_offset,
                                            StripDigiView out,
                                            int* aoffIdx,
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
            printf("offset should already set");
            break;
          }
          const uint_fast8_t newFirstStrip = data[(offset++) ^ 7];
          if (newFirstStrip < (firstStrip + inCluster)) {
            // LogDebug("FEDBuffer") << "First strip of new cluster is not greater than last strip of previous cluster. "
            //                       << "Last strip of previous cluster is " << uint16_t(firstStrip + inCluster) << ". "
            //                       << "First strip of new cluster is " << uint16_t(newFirstStrip) << ".";
            printf("StatusCode::UNORDERED_DATA");
            return StatusCode::UNORDERED_DATA;
          }
          firstStrip = newFirstStrip;
          nInCluster = data[(offset++) ^ 7];
          inCluster = 0;
          // out.stripId(++(*aoffIdx)) = 0xFFFF; // wonder what is the reason for these offsets
          // from my investigation, there is always a call at the begin and end of the cluster.
          // If this is a general property, then these two rows of the out StripDigiView could be saved for each unpackZSW call
          for (int i = 0; i < 2; ++i, ++(*aoffIdx)) {
            out.stripId(*aoffIdx) = 0xFFFF;
            out.adc(*aoffIdx) = 0;
          }
        }
        // assert(*aoffIdx!=23);
        out.channel(*aoffIdx) = chan;
        out.stripId(*aoffIdx) = stripStart + firstStrip + inCluster;
        out.adc(*aoffIdx) = getADC_W<num_words>(data, offset, bits_shift);
        (*aoffIdx)++;
        offset += num_words;
        ++inCluster;
        // printf("\t [stripID %i] [adc %i]\n", out.stripId(*idx-1), out.adc(*idx-1));
      }
      return StatusCode::SUCCESS;
    }

    // Generic implementation (for 10bit, essentially)
    template <uint_fast8_t num_bits>
    ALPAKA_FN_HOST_ACC StatusCode unpackZSB(int chan,
                                            const uint8_t* channel_data,
                                            uint16_t channel_length,
                                            uint32_t channel_offset,
                                            StripDigiView out,
                                            int* idx,
                                            uint8_t headerLength,
                                            uint16_t stripStart) {
      constexpr uint16_t mask = (1 << num_bits) - 1;
      if (channel_length & 0xF000) {
        // LogDebug("FEDBuffer") << "Channel length is invalid. Channel length is " << uint16_t(channel.length()) << ".";
        printf("[%i] | BAD_CHANNEL_LENGTH\n", *idx);
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
            printf("[%i] | UNORDERED_DATA - %i\t%i\t%i\n", *idx, newFirstStrip, firstStrip, inCluster);
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
        out.channel(*idx) = stripStart + firstStrip + inCluster;
        ++inCluster;
        if (bOffset == BITS_PER_BYTE) {
          bOffset = 0;
          ++wOffset;
        }
      }
      printf("[%i] | SUCCESS\n", *idx);
      return StatusCode::SUCCESS;
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint16_t readoutOrder(uint16_t physical_order) {
      return (4 * ((static_cast<uint16_t>((static_cast<float>(physical_order) / 8.0))) % 4) +
              static_cast<uint16_t>(static_cast<float>(physical_order) / 32.0) + 16 * (physical_order % 8));
    }
  }  // namespace detail

  namespace checks {
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isZeroSuppressed(
        FEDReadoutMode mode, bool legacy = false, FEDLegacyReadoutMode lmode = READOUT_MODE_LEGACY_INVALID) {
      if (!legacy) {
        switch (mode) {
          case READOUT_MODE_ZERO_SUPPRESSED_LITE10:
          case READOUT_MODE_ZERO_SUPPRESSED_LITE10_CMOVERRIDE:
          case READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT:
          case READOUT_MODE_PREMIX_RAW:
          case READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT_CMOVERRIDE:
          case READOUT_MODE_ZERO_SUPPRESSED_LITE8_CMOVERRIDE:
          case READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT:
          case READOUT_MODE_ZERO_SUPPRESSED:
          case READOUT_MODE_ZERO_SUPPRESSED_FAKE:
          case READOUT_MODE_ZERO_SUPPRESSED_LITE8:
          case READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT_CMOVERRIDE:
            return true;
            break;
          default:
            return false;
        }
      } else {
        switch (lmode) {
          case READOUT_MODE_LEGACY_ZERO_SUPPRESSED_REAL:
          case READOUT_MODE_LEGACY_ZERO_SUPPRESSED_FAKE:
          case READOUT_MODE_LEGACY_ZERO_SUPPRESSED_LITE_REAL:
          case READOUT_MODE_LEGACY_ZERO_SUPPRESSED_LITE_FAKE:
          case READOUT_MODE_LEGACY_PREMIX_RAW:
            return true;
          default:
            return false;
        }
      }
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isNonLiteZS(FEDReadoutMode mode,
                                                         bool legacy = false,
                                                         FEDLegacyReadoutMode lmode = READOUT_MODE_LEGACY_INVALID) {
      return (!legacy) ? (mode == READOUT_MODE_ZERO_SUPPRESSED || mode == READOUT_MODE_ZERO_SUPPRESSED_FAKE)
                       : (lmode == READOUT_MODE_LEGACY_ZERO_SUPPRESSED_REAL ||
                          lmode == READOUT_MODE_LEGACY_ZERO_SUPPRESSED_FAKE);
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isVirginRaw(FEDReadoutMode mode,
                                                         bool legacy = false,
                                                         FEDLegacyReadoutMode lmode = READOUT_MODE_LEGACY_INVALID) {
      return (!legacy) ? mode == READOUT_MODE_VIRGIN_RAW
                       : (lmode == READOUT_MODE_LEGACY_VIRGIN_RAW_REAL || lmode == READOUT_MODE_LEGACY_VIRGIN_RAW_FAKE);
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isProcessedRaw(FEDReadoutMode mode,
                                                            bool legacy = false,
                                                            FEDLegacyReadoutMode lmode = READOUT_MODE_LEGACY_INVALID) {
      return (!legacy) ? mode == READOUT_MODE_PROC_RAW
                       : (lmode == READOUT_MODE_LEGACY_PROC_RAW_REAL || lmode == READOUT_MODE_LEGACY_PROC_RAW_FAKE);
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isScopeMode(FEDReadoutMode mode,
                                                         bool legacy = false,
                                                         FEDLegacyReadoutMode lmode = READOUT_MODE_LEGACY_INVALID) {
      return (!legacy) ? mode == READOUT_MODE_SCOPE : lmode == READOUT_MODE_LEGACY_SCOPE;
    }
  }  // namespace checks

  namespace unpackers {
    ALPAKA_FN_HOST_ACC StatusCode unpackZeroSuppressed(int chanIdx,
                                                       const uint8_t* channel_data,
                                                       uint16_t channel_length,
                                                       uint32_t channel_offset,

                                                       StripDigiView out,
                                                       int* aoff,

                                                       uint16_t stripStart,
                                                       bool isNonLite,
                                                       FEDReadoutMode mode,
                                                       bool legacy = false,
                                                       FEDLegacyReadoutMode lmode = READOUT_MODE_LEGACY_INVALID,
                                                       uint8_t packetCode = 0) {
      if ((isNonLite && packetCode == PACKET_CODE_ZERO_SUPPRESSED10) ||
          ((!legacy) &&
           (mode == READOUT_MODE_ZERO_SUPPRESSED_LITE10 || mode == READOUT_MODE_ZERO_SUPPRESSED_LITE10_CMOVERRIDE))) {
        // printf("[%i] unpackZSB<10>\n", *aoff);
        return detail::unpackZSB<10>(
            chanIdx, channel_data, channel_length, channel_offset, out, aoff, (isNonLite ? 7 : 2), stripStart);
      } else if ((!legacy) ? mode == READOUT_MODE_PREMIX_RAW : lmode == READOUT_MODE_LEGACY_PREMIX_RAW) {
        // printf("[%i] unpackZSB<16>\n", *aoff);
        return detail::unpackZSW<16>(chanIdx, channel_data, channel_length, channel_offset, out, aoff, 7, stripStart);
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
                                       out,
                                       aoff,
                                       (isNonLite ? 7 : 2),
                                       stripStart,
                                       bits_shift);
        if (isNonLite && packetCode == 0 && StatusCode::SUCCESS == st) {
          // workaround for a pre-2015 bug in the packer: assume default ZS packing
          printf("[%i] ZERO_PACKET_CODE\n", *aoff);
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

  // Move the original constants to constexpr ()
  constexpr uint16_t get_FED_ID_MIN() { return sistrip::FED_ID_MIN; };
  constexpr uint16_t get_FEDCH_PER_FED() { return sistrip::FEDCH_PER_FED; };
  constexpr uint16_t get_STRIPS_PER_FEDCH() { return sistrip::STRIPS_PER_FEDCH; };
  constexpr uint16_t get_APVS_PER_FEDCH() { return sistrip::APVS_PER_FEDCH; };
  constexpr uint16_t get_APVS_PER_CHAN() { return sistrip::APVS_PER_CHAN; };
  constexpr uint16_t get_STRIPS_PER_APV() { return sistrip::STRIPS_PER_APV; };

  constexpr uint16_t FED_ID_MIN = get_FED_ID_MIN();
  constexpr uint16_t FEDCH_PER_FED = get_FEDCH_PER_FED();
  constexpr uint16_t STRIPS_PER_FEDCH = get_STRIPS_PER_FEDCH();
  constexpr uint16_t APVS_PER_FEDCH = get_APVS_PER_FEDCH();
  constexpr uint16_t APVS_PER_CHAN = get_APVS_PER_CHAN();
  constexpr uint16_t STRIPS_PER_APV = get_STRIPS_PER_APV();

  constexpr uint16_t invalidStrip = std::numeric_limits<uint16_t>::max();
  constexpr uint16_t invalidFed = std::numeric_limits<uint16_t>::max();

  constexpr uint16_t badBit = (1 << 15);

  constexpr int32_t kMaxSeedStrips = 200000;
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
    ALPAKA_FN_HOST_ACC void operator()(TAcc const& acc,
                                       bool legacy,
                                       FEDLegacyReadoutMode lmode,
                                       //
                                       SiStripMappingConstView mapping,
                                       SiStripClusterizerConditionsDetToFedsConstView DetToFeds,
                                       SiStripClusterizerConditionsData_fedchConstView Data_fedch,
                                       SiStripClusterizerConditionsData_stripConstView Data_strip,
                                       SiStripClusterizerConditionsData_apvConstView Data_apv,
                                       StripDigiView stripDataObj) const {
      // Loop over the FEDChannel collection to be digitized
      for (auto chan : uniform_elements(acc, mapping.metadata().size())) {
        const auto fedID = mapping.fedID(chan);
        const auto fedCH = mapping.fedCh(chan);
        // const auto detID = mapping.detID(chan);

        if (fedID == invalidFed)
          continue;  // reject strips which are in the conditions but not in the fed data collection

        const auto ipair = Data_fedch.iPair_(channelIndex(fedID, fedCH));
        int ipoff = STRIPS_PER_FEDCH * ipair;

        const unsigned char* channel_data = mapping.input(chan);
        const short unsigned channel_len = mapping.length(chan);
        const long unsigned channel_offset = mapping.inoff(chan);

        FEDReadoutMode mode = mapping.readoutMode(chan);
        const unsigned pCode = mapping.packetCode(chan);
        bool isNonLite = fedchannelunpacker::checks::isNonLiteZS(mode, legacy, lmode);

        int aoff = mapping.offset(chan);

        // unpackZeroSuppressed(const unsigned char* const&, const short unsigned int&, const long unsigned int&, sistripclusterizer::StripDigiView&, long unsigned int*, int, bool&, sistrip::FEDReadoutMode&, bool&, sistrip::FEDLegacyReadoutMode&, const unsigned char&)
        auto retCode = fedchannelunpacker::unpackers::unpackZeroSuppressed(chan,
                                                                           channel_data,
                                                                           channel_len,
                                                                           channel_offset,
                                                                           stripDataObj,
                                                                           &aoff,
                                                                           ipoff,
                                                                           isNonLite,
                                                                           mode,
                                                                           legacy,
                                                                           lmode,
                                                                           pCode);
        if ((int)retCode != 0) {
          printf("[%i] [fedID %i] [fedCH %i] - Returned %i\n", chan, fedID, fedCH, (int)retCode);
        }

        // if (channel_data != nullptr && channel_len > 0) {
        //   auto aoff = mapping.offset(chan);
        //   auto choff = mapping.inoff(chan);
        //   const auto end = choff + channel_len;

        //   while (choff < end) {
        //     auto stripIndex = channel_data[(choff++) ^ 7] + ipoff;
        //     const auto groupLength = channel_data[(choff++) ^ 7];

        //     // initialize as invalid strip
        //     for (auto i = 0; i < 2; ++i) {
        //       stripDataObj.stripId(aoff) = 0xFFFF;
        //       stripDataObj.adc(aoff++) = 0;
        //     }

        //     for (auto i = 0; i < groupLength; ++i) {
        //       stripDataObj.stripId(aoff) = stripIndex++;
        //       stripDataObj.channel(aoff) = chan;
        //       stripDataObj.adc(aoff++) = channel_data[(choff++) ^ 7];
        //     }
        //   }
        // }  // choff < end
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
          // clusterDataObj.seedStripsNCMask(chan) = static_cast<uint8_t>(seedThreshold); // debugging
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

  class SiStripRawToClusterAlgoKernel_setStripIndex {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  StripDigiConstView stripDataObj,
                                  StripClustersAuxView clusterDataObj) const {
      // Loop over the strips
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
      const auto nSeedStripsNC = (kMaxSeedStrips < clusterDataObj.prefixSeedStripsNCMask(nStrips - 1))
                                     ? kMaxSeedStrips
                                     : clusterDataObj.prefixSeedStripsNCMask(nStrips - 1);
      auto channels = stripDataObj.channel();
      auto stripId = stripDataObj.stripId();
      auto adc = stripDataObj.adc();
      const auto channelThreshold = clusterDataObj.channelThreshold();
      const auto maxSequentialHoles = clusterDataObj.maxSequentialHoles();
      int clusterSizeLimit = clusterDataObj.clusterSizeLimit();
      const auto clusterThresholdSquared = clusterDataObj.clusterThresholdSquared();

      // Loop over only the non-contiguous strips (flagged in setStripIndex)
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
      const auto nClusters = clusters.nClusters();
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

      for (auto i : uniform_elements(acc, nClusters)) {
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
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

// kernels launchers
namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace cms::alpakatools;
  using namespace sistripclusterizer;

  SiStripRawToClusterAlgo::SiStripRawToClusterAlgo(const edm::ParameterSet& unpackPar,
                                                   const edm::ParameterSet& clustPar)
      : isLegacyUnpacker_(unpackPar.getParameter<bool>("LegacyUnpacker")),
        channelThreshold_(clustPar.getParameter<double>("ChannelThreshold")),
        seedThreshold_(clustPar.getParameter<double>("SeedThreshold")),
        clusterThresholdSquared_(std::pow(clustPar.getParameter<double>("ClusterThreshold"), 2.0f)),
        maxSequentialHoles_(clustPar.getParameter<unsigned>("MaxSequentialHoles")),
        maxSequentialBad_(clustPar.getParameter<unsigned>("MaxSequentialBad")),
        maxAdjacentBad_(clustPar.getParameter<unsigned>("MaxAdjacentBad")),
        maxClusterSize_(clustPar.getParameter<unsigned>("MaxClusterSize")),
        minGoodCharge_(clusterChargeCut(clustPar)) {
    // Make sure the module does not start with features not implemented yet
    if (maxClusterSize_ > 32) {
      throw cms::Exception("SiStripRawToClstAlg", "MaxClusterSize must be <= 32");
    }
    if (isLegacyUnpacker_) {
      throw cms::Exception("SiStripRawToClstAlg", "Legacy unpacking not supported yet");
    }
  }

  void SiStripRawToClusterAlgo::initialize(Queue& queue, int n_strips) {
    assert(n_strips >= 0);
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
    // I wonder if there is an helper function which automatically optimize this based on the accelerator properties
    uint32_t divider = 128;
    // use as many groups as needed to cover the whole problem
    auto nStrips = mapping->metadata().size();
    uint32_t groups = divide_up_by(nStrips, divider);

    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(groups, divider);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripRawToClusterAlgoKernel_unpacker{},
                        isLegacyUnpacker_,
                        legacyUnpackerROmode_,
                        //
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
                                // std::cout << "[SiStripRawToClusterAlgo::unpackStrips] Dumping channel_...\n"
                                << "i\tadc\tchan\tstripId\n";
    for (int i = 0; i < sClustersAux_host->metadata().size(); ++i) {
      if ((i % 1000 == 0 and sClustersAux_host->stripId(i) != invalidStrip) or (i <= 100) or
          (i > (sClustersAux_host->metadata().size() - 100))) {
        LogDebug("SiStripUnpkDigi")
            << i << "\t" << (int)sClustersAux_host->adc(i)
            << " "
            // std::cout << i << "\t" << (int)sClustersAux_host->adc(i) << "\t" << (int)sClustersAux_host->channel(i) << "\t"
            << (int)sClustersAux_host->stripId(i) << "\n";
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
    uint32_t divider = 128;
    auto nStrips = mapping->metadata().size();
    uint32_t groups = divide_up_by(nStrips, divider);

    auto workDiv = make_workdiv<Acc1D>(groups, divider);

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
    // From example in HeterogeneousCore/AlpakaInterface/test/alpaka/testPrefixScan.dev.cc
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

  std::unique_ptr<SiStripClustersDevice> SiStripRawToClusterAlgo::makeClusters(
      Queue& queue, SiStripMappingDevice const& mapping, SiStripClusterizerConditionsDevice const& conditions) {
    // The maximum number of clusters is set to kMaxSeedStrips
    auto clusters_d = std::make_unique<SiStripClustersDevice>(kMaxSeedStrips, queue);
    // The number of seed over which to loop for clusters is the min between the number of strips and the kMaxSeeds
    const auto nStrips = clustersAux_d_->view().metadata().size();
    const int nSeeds = std::min(kMaxSeedStrips, nStrips);

    uint32_t divider = 128;
    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(nSeeds, divider);
    auto workDiv = make_workdiv<Acc1D>(groups, divider);

    // Three-threshold clusterization algo
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripRawToClusterAlgoKernel_findLeftRightBoundaryGPU{},
                        mapping.const_view(),
                        conditions.const_view<SiStripClusterizerConditionsData_stripSoA>(),
                        clustersAux_d_->const_view(),
                        clustersAux_d_->view<StripClustersAuxSoA>(),
                        clusters_d->view());

    // Apply the conditions
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripRawToClusterAlgoKernel_checkClusterConditionGPU{},
                        mapping.const_view(),
                        conditions.const_view<SiStripClusterizerConditionsData_fedchSoA>(),
                        conditions.const_view<SiStripClusterizerConditionsData_apvSoA>(),
                        clustersAux_d_->const_view(),
                        clustersAux_d_->view<StripClustersAuxSoA>(),
                        clusters_d->view());

#if defined(EDM_ML_DEBUG) && defined(SUPERDETAILS)
    checkClusters(queue, clusters_d.get());
#endif

    return clusters_d;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#ifdef EDM_ML_DEBUG  // debug functions
namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
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

  void SiStripRawToClusterAlgo::checkClusters(Queue& queue, SiStripClustersDevice* clusters_d) const {
    // Copy the clusted on the host
    SiStripClustersHost clusters_h = SiStripClustersHost(clusters_d->view().metadata().size(), queue);
    alpaka::memcpy(queue, clusters_h.buffer(), clusters_d->buffer());
    alpaka::wait(queue);

    const int clustersPrealloc = clusters_d->view().metadata().size();
    const int clustersN = clusters_h->nClusters();
    LogDebug("SiStripClusChk") << "Clusters report\n"
                               << "Pre-allocated:\t" << clustersPrealloc << "\tProduced:\t" << clustersN << "\n";

    LogDebug("SiStripClusChk") << "   -----  Small cluster dump BEGIN -----   \n";
    LogDebug("SiStripClusChk") << "i\tcIdx\tcSz\tcDetId\t1Strip\ttrueC\tbary\tchg\t - clusterADCs\n";
    // Print the result
    for (int i = 0; i < clustersN; ++i) {
      if (i > 100 && i < (clustersN - 1000)) {
        continue;
      } else {
        LogDebug("SiStripClusChk") << i << "\t" << clusters_h->clusterIndex(i) << "\t" << clusters_h->clusterSize(i)
                                   << "\t" << clusters_h->clusterDetId(i) << "\t" << clusters_h->charge(i) << "\t"
                                   << clusters_h->firstStrip(i) << "\t" << clusters_h->trueCluster(i) << "\t"
                                   << clusters_h->barycenter(i) << "\t - ";
        for (unsigned int j = 0; j < clusters_h->clusterSize(i); ++j) {
          LogDebug("") << j << ":" << (int)(clusters_h->clusterADCs(i)[j]) << "  ";
        }
      }
      LogDebug("") << "\n";
    }
    LogDebug("SiStripClusChk") << "   -----  Small cluster dump END   -----   \n";
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip
#endif  // debug functions
