#ifndef EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H
#define EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H

#include <string>
#include <vector>
#include <memory>
#include <ostream>
#include <cstring>
#include <cmath>
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include <cstdint>

namespace sistrip {
  constexpr uint16_t BITS_PER_BYTE = 8;

  //
  // Class definitions
  //

  //class representing standard (non-spy channel) FED buffers
  class FEDBuffer final : public FEDBufferBase {
  public:
    /**
     * constructor from a FEDRawData buffer
     *
     * The sistrip::preconstructCheckFEDBuffer() method should be used
     * (with the same value of allowBadBuffer) to check the validity of
     * fedBuffer before constructing a sistrip::FEDBuffer.
     * If allowBadBuffer is set to true, the initialization proceeds
     * even if the event format is not recognized.
     * To initialize also the channel information, the FEDBuffer::findChannels()
     * method should be called as well, and its return status checked
     * (unless bad buffers, with an unrecognized event format or channel lengths
     * that do not make sense, should also be included).
     *
     * @see sistrip::preconstructCheckFEDBuffer() sistrip::FEDBuffer::findChannels()
     */
    explicit FEDBuffer(const FEDRawData& fedBuffer, const bool allowBadBuffer = false);
    ~FEDBuffer() override {}

    /**
     * Read the channel lengths from the payload
     *
     * This method should be called to after the constructor
     * (and should not be called more than once for the same sistrip::FEDBuffer).
     * In case any check fails, a value different from sistrip::FEDBufferStatusCode::SUCCESS
     * is returned, and detailed information printed to LogDebug("FEDBuffer"), if relevant.
     *
     * @see sistrip::FEDBuffer::FEDBuffer()
     */
    FEDBufferStatusCode findChannels();

    void print(std::ostream& os) const override;
    const FEDFEHeader* feHeader() const;
    //check that a FE unit is enabled, has a good majority address and, if in full debug mode, that it is present
    bool feGood(const uint8_t internalFEUnitNum) const;
    bool feGoodWithoutAPVEmulatorCheck(const uint8_t internalFEUnitNum) const;
    //check that a FE unit is present in the data.
    //The high order byte of the FEDStatus register in the tracker special header is used in APV error mode.
    //The FE length from the full debug header is used in full debug mode.
    bool fePresent(uint8_t internalFEUnitNum) const;
    //check that a channel is present in data, found, on a good FE unit and has no errors flagged in status bits
    using sistrip::FEDBufferBase::channelGood;
    bool channelGood(const uint8_t internalFEDannelNum, const bool doAPVeCheck) const;
    void setLegacyMode(bool legacy) { legacyUnpacker_ = legacy; }

    //functions to check buffer. All return true if there is no problem.
    //minimum checks to do before using buffer
    using sistrip::FEDBufferBase::doChecks;
    bool doChecks(bool doCRC) const;

    //additional checks to check for corrupt buffers
    //check channel lengths fit inside to buffer length
    bool checkChannelLengths() const;
    //check that channel lengths add up to buffer length (this does the previous check as well)
    bool checkChannelLengthsMatchBufferLength() const;
    //check channel packet codes match readout mode
    bool checkChannelPacketCodes() const;
    //check FE unit lengths in FULL DEBUG header match the lengths of their channels
    bool checkFEUnitLengths() const;
    //check FE unit APV addresses in FULL DEBUG header are equal to the APVe address if the majority was good
    bool checkFEUnitAPVAddresses() const;
    //do all corrupt buffer checks
    virtual bool doCorruptBufferChecks() const;

    //check that there are no errors in channel, APV or FEUnit status bits
    //these are done by channelGood(). Channels with bad status bits may be disabled so bad status bits do not usually indicate an error
    bool checkStatusBits(const uint8_t internalFEDChannelNum) const;
    bool checkStatusBits(const uint8_t internalFEUnitNum, const uint8_t internalChannelNum) const;
    //same but for all channels on enabled FE units
    bool checkAllChannelStatusBits() const;

    //check that all FE unit payloads are present
    bool checkFEPayloadsPresent() const;

    //print a summary of all checks
    std::string checkSummary() const override;

  private:
    uint8_t nFEUnitsPresent() const;
    inline uint8_t getCorrectPacketCode() const { return packetCode(legacyUnpacker_); }
    uint16_t calculateFEUnitLength(const uint8_t internalFEUnitNumber) const;
    std::unique_ptr<FEDFEHeader> feHeader_;
    const uint8_t* payloadPointer_;
    uint16_t payloadLength_;
    uint8_t validChannels_;
    bool fePresent_[FEUNITS_PER_FED];
    bool legacyUnpacker_ = false;
  };

  //
  // Inline function definitions
  //

  /**
   * Check if a FEDRawData object satisfies the requirements for constructing a sistrip::FEDBuffer
   *
   * These are:
   *   - those from sistrip::preconstructCheckFEDBufferBase()
   *   - the readout mode should not be sistrip::READOUT_MODE_SPY
   *   - (unless allowBadBuffers is true) the header type should not be sistrip::HEADER_TYPE_INVALID or HEADER_TYPE_NONE
   *
   * In case any check fails, a value different from sistrip::FEDBufferStatusCode::SUCCESS
   * is returned, and detailed information printed to LogDebug("FEDBuffer"), if relevant.
   *
   * @see sistrip::preconstructCheckFEDBufferBase()
   */
  inline FEDBufferStatusCode preconstructCheckFEDBuffer(const FEDRawData& fedBuffer, bool allowBadBuffer = false) {
    const auto st_base = preconstructCheckFEDBufferBase(fedBuffer, !allowBadBuffer);
    if (FEDBufferStatusCode::SUCCESS != st_base)
      return st_base;
    const TrackerSpecialHeader hdr{fedBuffer.data() + 8};
    const auto hdr_type = hdr.headerType();
    if ((!allowBadBuffer) && ((hdr_type == sistrip::HEADER_TYPE_INVALID) || (hdr_type == sistrip::HEADER_TYPE_NONE))) {
#ifdef EDM_ML_DEBUG
      std::ostringstream msg;
      msg << "Header type is invalid. Header type nibble is ";
      const auto headerTypeNibble = hdr.headerTypeNibble();
      printHex(&headerTypeNibble, 1, msg);
      LogDebug("FEDBuffer") << msg.str();
#endif
      return FEDBufferStatusCode::WRONG_HEADERTYPE;
    }
    if (READOUT_MODE_SPY == hdr.readoutMode())
      return FEDBufferStatusCode::EXPECT_NOT_SPY;
    return FEDBufferStatusCode::SUCCESS;
  }

  //FEDBuffer

  inline const FEDFEHeader* FEDBuffer::feHeader() const { return feHeader_.get(); }

  inline bool FEDBuffer::channelGood(const uint8_t internalFEDChannelNum, const bool doAPVeCheck) const {
    return ((internalFEDChannelNum < validChannels_) &&
            ((doAPVeCheck && feGood(internalFEDChannelNum / FEDCH_PER_FEUNIT)) ||
             (!doAPVeCheck && feGoodWithoutAPVEmulatorCheck(internalFEDChannelNum / FEDCH_PER_FEUNIT))) &&
            (this->readoutMode() == sistrip::READOUT_MODE_SCOPE || checkStatusBits(internalFEDChannelNum)));
  }

  inline bool FEDBuffer::feGood(const uint8_t internalFEUnitNum) const {
    return (!majorityAddressErrorForFEUnit(internalFEUnitNum) && !feOverflow(internalFEUnitNum) &&
            fePresent(internalFEUnitNum));
  }

  inline bool FEDBuffer::feGoodWithoutAPVEmulatorCheck(const uint8_t internalFEUnitNum) const {
    return (!feOverflow(internalFEUnitNum) && fePresent(internalFEUnitNum));
  }

  inline bool FEDBuffer::fePresent(uint8_t internalFEUnitNum) const { return fePresent_[internalFEUnitNum]; }

  inline bool FEDBuffer::checkStatusBits(const uint8_t internalFEDChannelNum) const {
    return feHeader_->checkChannelStatusBits(internalFEDChannelNum);
  }

  inline bool FEDBuffer::checkStatusBits(const uint8_t internalFEUnitNum, const uint8_t internalChannelNum) const {
    return checkStatusBits(internalFEDChannelNum(internalFEUnitNum, internalChannelNum));
  }

  inline bool FEDBuffer::doChecks(bool doCRC) const {
    //check that all channels were unpacked properly
    return (validChannels_ == FEDCH_PER_FED) &&
           //do checks from base class
           (FEDBufferBase::doChecks()) &&
           // check crc if required
           (!doCRC || checkCRC());
  }

  namespace fedchannelunpacker {
    enum class StatusCode { SUCCESS = 0, BAD_CHANNEL_LENGTH, UNORDERED_DATA, BAD_PACKET_CODE, ZERO_PACKET_CODE };

    namespace detail {

      template <uint8_t num_words>
      uint16_t getADC_W(const uint8_t* data, uint_fast16_t offset, uint8_t bits_shift) {
        // get ADC from one or two bytes (at most 10 bits), and shift if needed
        return (data[offset ^ 7] + (num_words == 2 ? ((data[(offset + 1) ^ 7] & 0x03) << 8) : 0)) << bits_shift;
      }

      template <uint16_t mask>
      uint16_t getADC_B2(const uint8_t* data, uint_fast16_t wOffset, uint_fast8_t bOffset) {
        // get ADC from two bytes, from wOffset until bOffset bits from the next byte (maximum decided by mask)
        return (((data[wOffset ^ 7]) << bOffset) + (data[(wOffset + 1) ^ 7] >> (BITS_PER_BYTE - bOffset))) & mask;
      }
      template <uint16_t mask>
      uint16_t getADC_B1(const uint8_t* data, uint_fast16_t wOffset, uint_fast8_t bOffset) {
        // get ADC from one byte, until bOffset into the byte at wOffset (maximum decided by mask)
        return (data[wOffset ^ 7] >> (BITS_PER_BYTE - bOffset)) & mask;
      }

      // Unpack Raw with ADCs in whole 8-bit words (8bit and 10-in-16bit)
      template <uint8_t num_bits, typename OUT>
      StatusCode unpackRawW(const FEDChannel& channel, OUT&& out, uint8_t bits_shift = 0) {
        constexpr auto num_words = num_bits / 8;
        static_assert(((num_bits % 8) == 0) && (num_words > 0) && (num_words < 3));
        if ((num_words > 1) && ((channel.length() - 3) % num_words)) {
          LogDebug("FEDBuffer") << "Channel length is invalid. Raw channels have 3 header bytes and " << num_words
                                << " bytes per sample. "
                                << "Channel length is " << uint16_t(channel.length()) << ".";
          return StatusCode::BAD_CHANNEL_LENGTH;
        }
        const uint8_t* const data = channel.data();
        const uint_fast16_t end = channel.offset() + channel.length();
        for (uint_fast16_t offset = channel.offset() + 3; offset != end; offset += num_words) {
          *out++ = SiStripRawDigi(getADC_W<num_words>(data, offset, bits_shift));
        }
        return StatusCode::SUCCESS;
      }

      // Generic implementation for non-whole words (10bit, essentially)
      template <uint_fast8_t num_bits, typename OUT>
      StatusCode unpackRawB(const FEDChannel& channel, OUT&& out) {
        static_assert(num_bits <= 16, "Word length must be between 0 and 16.");
        if (channel.length() & 0xF000) {
          LogDebug("FEDBuffer") << "Channel length is invalid. Channel length is " << uint16_t(channel.length()) << ".";
          return StatusCode::BAD_CHANNEL_LENGTH;
        }
        constexpr uint16_t mask = (1 << num_bits) - 1;
        const uint8_t* const data = channel.data();
        const uint_fast16_t chEnd = channel.offset() + channel.length();
        uint_fast16_t wOffset = channel.offset() + 3;
        uint_fast8_t bOffset = 0;
        while (((wOffset + 1) < chEnd) || ((chEnd - wOffset) * BITS_PER_BYTE - bOffset >= num_bits)) {
          bOffset += num_bits;
          if ((num_bits > BITS_PER_BYTE) || (bOffset > BITS_PER_BYTE)) {
            bOffset -= BITS_PER_BYTE;
            **out++ = SiStripRawDigi(getADC_B2<mask>(data, wOffset, bOffset));
            ++wOffset;
          } else {
            **out++ = SiStripRawDigi(getADC_B1<mask>(data, wOffset, bOffset));
          }
          if (bOffset == BITS_PER_BYTE) {
            bOffset = 0;
            ++wOffset;
          }
        }
        return StatusCode::SUCCESS;
      }

      template <uint8_t num_bits, typename OUT>
      StatusCode unpackZSW(
          const FEDChannel& channel, OUT&& out, uint8_t headerLength, uint16_t stripStart, uint8_t bits_shift = 0) {
        constexpr auto num_words = num_bits / 8;
        static_assert(((num_bits % 8) == 0) && (num_words > 0) && (num_words < 3));
        if (channel.length() & 0xF000) {
          LogDebug("FEDBuffer") << "Channel length is invalid. Channel length is " << uint16_t(channel.length()) << ".";
          return StatusCode::BAD_CHANNEL_LENGTH;
        }
        const uint8_t* const data = channel.data();
        uint_fast16_t offset = channel.offset() + headerLength;  // header is 2 (lite) or 7
        uint_fast8_t firstStrip{0}, nInCluster{0}, inCluster{0};
        const uint_fast16_t end = channel.offset() + channel.length();
        while (offset != end) {
          if (inCluster == nInCluster) {
            if (offset + 2 >= end) {
              // offset should already be at end then (empty cluster)
              break;
            }
            const uint_fast8_t newFirstStrip = data[(offset++) ^ 7];
            if (newFirstStrip < (firstStrip + inCluster)) {
              LogDebug("FEDBuffer") << "First strip of new cluster is not greater than last strip of previous cluster. "
                                    << "Last strip of previous cluster is " << uint16_t(firstStrip + inCluster) << ". "
                                    << "First strip of new cluster is " << uint16_t(newFirstStrip) << ".";
              return StatusCode::UNORDERED_DATA;
            }
            firstStrip = newFirstStrip;
            nInCluster = data[(offset++) ^ 7];
            inCluster = 0;
          }
          *out++ = SiStripDigi(stripStart + firstStrip + inCluster, getADC_W<num_words>(data, offset, bits_shift));
          offset += num_words;
          ++inCluster;
        }
        return StatusCode::SUCCESS;
      }

      // Generic implementation (for 10bit, essentially)
      template <uint_fast8_t num_bits, typename OUT>
      StatusCode unpackZSB(const FEDChannel& channel, OUT&& out, uint8_t headerLength, uint16_t stripStart) {
        constexpr uint16_t mask = (1 << num_bits) - 1;
        if (channel.length() & 0xF000) {
          LogDebug("FEDBuffer") << "Channel length is invalid. Channel length is " << uint16_t(channel.length()) << ".";
          return StatusCode::BAD_CHANNEL_LENGTH;
        }
        const uint8_t* const data = channel.data();
        uint_fast16_t wOffset = channel.offset() + headerLength;  // header is 2 (lite) or 7
        uint_fast8_t bOffset{0}, firstStrip{0}, nInCluster{0}, inCluster{0};
        const uint_fast16_t chEnd = channel.offset() + channel.length();
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
            const uint_fast8_t newFirstStrip = data[(wOffset++) ^ 7];
            if (newFirstStrip < (firstStrip + inCluster)) {
              LogDebug("FEDBuffer") << "First strip of new cluster is not greater than last strip of previous cluster. "
                                    << "Last strip of previous cluster is " << uint16_t(firstStrip + inCluster) << ". "
                                    << "First strip of new cluster is " << uint16_t(newFirstStrip) << ".";
              return StatusCode::UNORDERED_DATA;
            }
            firstStrip = newFirstStrip;
            nInCluster = data[(wOffset++) ^ 7];
            inCluster = 0;
            bOffset = 0;
          }
          bOffset += num_bits;
          if ((num_bits > BITS_PER_BYTE) || (bOffset > BITS_PER_BYTE)) {
            bOffset -= BITS_PER_BYTE;
            *out++ = SiStripDigi(stripStart + firstStrip + inCluster, getADC_B2<mask>(data, wOffset, bOffset));
            ++wOffset;
          } else {
            *out++ = SiStripDigi(stripStart + firstStrip + inCluster, getADC_B1<mask>(data, wOffset, bOffset));
          }
          ++inCluster;
          if (bOffset == BITS_PER_BYTE) {
            bOffset = 0;
            ++wOffset;
          }
        }
        return StatusCode::SUCCESS;
      }

      inline uint16_t readoutOrder(uint16_t physical_order) {
        return (4 * ((static_cast<uint16_t>((static_cast<float>(physical_order) / 8.0))) % 4) +
                static_cast<uint16_t>(static_cast<float>(physical_order) / 32.0) + 16 * (physical_order % 8));
      }
    };  // namespace detail

    inline bool isZeroSuppressed(FEDReadoutMode mode,
                                 bool legacy = false,
                                 FEDLegacyReadoutMode lmode = READOUT_MODE_LEGACY_INVALID) {
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
    inline bool isNonLiteZS(FEDReadoutMode mode,
                            bool legacy = false,
                            FEDLegacyReadoutMode lmode = READOUT_MODE_LEGACY_INVALID) {
      return (!legacy) ? (mode == READOUT_MODE_ZERO_SUPPRESSED || mode == READOUT_MODE_ZERO_SUPPRESSED_FAKE)
                       : (lmode == READOUT_MODE_LEGACY_ZERO_SUPPRESSED_REAL ||
                          lmode == READOUT_MODE_LEGACY_ZERO_SUPPRESSED_FAKE);
    }
    inline bool isVirginRaw(FEDReadoutMode mode,
                            bool legacy = false,
                            FEDLegacyReadoutMode lmode = READOUT_MODE_LEGACY_INVALID) {
      return (!legacy) ? mode == READOUT_MODE_VIRGIN_RAW
                       : (lmode == READOUT_MODE_LEGACY_VIRGIN_RAW_REAL || lmode == READOUT_MODE_LEGACY_VIRGIN_RAW_FAKE);
    }
    inline bool isProcessedRaw(FEDReadoutMode mode,
                               bool legacy = false,
                               FEDLegacyReadoutMode lmode = READOUT_MODE_LEGACY_INVALID) {
      return (!legacy) ? mode == READOUT_MODE_PROC_RAW
                       : (lmode == READOUT_MODE_LEGACY_PROC_RAW_REAL || lmode == READOUT_MODE_LEGACY_PROC_RAW_FAKE);
    }
    inline bool isScopeMode(FEDReadoutMode mode,
                            bool legacy = false,
                            FEDLegacyReadoutMode lmode = READOUT_MODE_LEGACY_INVALID) {
      return (!legacy) ? mode == READOUT_MODE_SCOPE : lmode == READOUT_MODE_LEGACY_SCOPE;
    }

    template <typename OUT>
    StatusCode unpackScope(const FEDChannel& channel, OUT&& out) {
      return detail::unpackRawW<16>(channel, out);
    }
    template <typename OUT>
    StatusCode unpackProcessedRaw(const FEDChannel& channel, OUT&& out) {
      return detail::unpackRawW<16>(channel, out);
    }

    template <typename OUT>
    StatusCode unpackVirginRaw(const FEDChannel& channel, OUT&& out, uint8_t packetCode) {
      std::vector<SiStripRawDigi> samples;
      auto st = StatusCode::SUCCESS;
      if (PACKET_CODE_VIRGIN_RAW == packetCode) {
        samples.reserve((channel.length() - 3) / 2);
        st = detail::unpackRawW<16>(channel, std::back_inserter(samples));
      } else if (PACKET_CODE_VIRGIN_RAW10 == packetCode) {
        samples.reserve((channel.length() - 3) * 10 / 8);
        st = detail::unpackRawB<10>(channel, std::back_inserter(samples));
      } else if (PACKET_CODE_VIRGIN_RAW8_BOTBOT == packetCode || PACKET_CODE_VIRGIN_RAW8_TOPBOT == packetCode) {
        samples.reserve(channel.length() - 3);
        st = detail::unpackRawW<8>(
            channel, std::back_inserter(samples), (PACKET_CODE_VIRGIN_RAW8_BOTBOT == packetCode ? 2 : 1));
      }
      if (!samples.empty()) {  // reorder
        for (uint_fast16_t i{0}; i != samples.size(); ++i) {
          const auto physical = i % 128;
          const auto readout = (detail::readoutOrder(physical) * 2  // convert index from physical to readout order
                                + (i >= 128 ? 1 : 0));              // un-multiplex data
          *out++ = samples[readout];
        }
      }
      return st;
    }
    template <typename OUT>
    StatusCode unpackZeroSuppressed(const FEDChannel& channel,
                                    OUT&& out,
                                    uint16_t stripStart,
                                    bool isNonLite,
                                    FEDReadoutMode mode,
                                    bool legacy = false,
                                    FEDLegacyReadoutMode lmode = READOUT_MODE_LEGACY_INVALID,
                                    uint8_t packetCode = 0) {
      if ((isNonLite && packetCode == PACKET_CODE_ZERO_SUPPRESSED10) ||
          ((!legacy) &&
           (mode == READOUT_MODE_ZERO_SUPPRESSED_LITE10 || mode == READOUT_MODE_ZERO_SUPPRESSED_LITE10_CMOVERRIDE))) {
        return detail::unpackZSB<10>(channel, out, (isNonLite ? 7 : 2), stripStart);
      } else if ((!legacy) ? mode == READOUT_MODE_PREMIX_RAW : lmode == READOUT_MODE_LEGACY_PREMIX_RAW) {
        return detail::unpackZSW<16>(channel, out, 7, stripStart);
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
        auto st = detail::unpackZSW<8>(channel, out, (isNonLite ? 7 : 2), stripStart, bits_shift);
        if (isNonLite && packetCode == 0 && StatusCode::SUCCESS == st) {
          // workaround for a pre-2015 bug in the packer: assume default ZS packing
          return StatusCode::ZERO_PACKET_CODE;
        }
        return st;
      }
    }
  };  // namespace fedchannelunpacker
  std::string toString(fedchannelunpacker::StatusCode status);
}  // namespace sistrip

#endif  //ndef EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H
