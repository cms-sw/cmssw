#include "EventFilter/HGCalRawToDigi/interface/HGCalUnpacker.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDPacketInfoHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <array>

using namespace hgcal;

uint8_t HGCalUnpacker::parseFEDData(unsigned fedId,
                                    const FEDRawData& fed_data,
                                    const HGCalMappingModuleIndexer& moduleIndexer,
                                    const HGCalConfiguration& config,
                                    hgcaldigi::HGCalDigiHost& digis,
                                    hgcaldigi::HGCalECONDPacketInfoHost& econdPacketInfo,
                                    bool headerOnlyMode) {
  // ReadoutSequence object for this FED
  const auto& fedReadoutSequence = moduleIndexer.getFEDReadoutSequences()[fedId];
  // Configuration object for this FED
  const auto& fedConfig = config.feds[fedId];

  // helper functions
  auto to_32b_words = [](const uint64_t* ptr_64b) {
    auto* ptr_32b = reinterpret_cast<const uint32_t*>(ptr_64b);
    return std::array<uint32_t, 2>{{ptr_32b[1], ptr_32b[0]}};
  };

  auto to_econd_payload = [](const uint64_t* ptr_64b, uint64_t payload_length) -> std::vector<uint32_t> {
    std::vector<uint32_t> payload(payload_length, 0);
    auto* ptr_32b = reinterpret_cast<const uint32_t*>(ptr_64b);
    for (unsigned i = 0; i < payload_length; ++i) {
      payload[i] = ptr_32b[(i % 2 == 0) ? i + 1 : i - 1];
    }
    return payload;
  };

  // Endianness assumption
  // From 32-bit word(ECOND) to 64-bit word(capture block): little endianness
  // Others: big endianness
  const auto* const header = reinterpret_cast<const uint64_t*>(fed_data.data());
  const auto* const trailer = reinterpret_cast<const uint64_t*>(fed_data.data() + fed_data.size());
  LogDebug("[HGCalUnpacker]") << "fedId = " << fedId << " nwords (64b) = " << std::distance(header, trailer);

  const auto* ptr = header;
  for (unsigned iword = 0; ptr < trailer; ++iword) {
    LogDebug("[HGCalUnpacker]") << std::setw(8) << iword << ": 0x" << std::hex << std::setfill('0') << std::setw(16)
                                << *ptr << " (" << std::setfill('0') << std::setw(8)
                                << *(reinterpret_cast<const uint32_t*>(ptr) + 1) << " " << std::setfill('0')
                                << std::setw(8) << *reinterpret_cast<const uint32_t*>(ptr) << ")" << std::dec;
    ++ptr;
  }

  LogDebug("[HGCalUnpacker]") << "@@@\n";
  ptr = header;
  // check SLink header (128b)
  // sanity check
  auto slink_header = *(ptr + 1);
  if (((slink_header >> (BACKEND_FRAME::SLINK_BOE_POS + 32)) & BACKEND_FRAME::SLINK_BOE_MASK) !=
      fedConfig.slinkHeaderMarker) {
    uint32_t ECONDdenseIdx = moduleIndexer.getIndexForModule(fedId, 0);
    econdPacketInfo.view()[ECONDdenseIdx].exception() = 1;
    econdPacketInfo.view()[ECONDdenseIdx].location() = 0;
    edm::LogWarning("[HGCalUnpacker]") << "Expected a S-Link header (BOE: 0x" << std::hex << fedConfig.slinkHeaderMarker
                                       << "), got 0x" << std::hex
                                       << ((slink_header >> (BACKEND_FRAME::SLINK_BOE_POS + 32)) &
                                           BACKEND_FRAME::SLINK_BOE_MASK)
                                       << " from " << slink_header << ".";
    return UNPACKER_STAT::WrongSLinkHeader;
  }

  ptr += 2;
  // counter for the global index of ECON-D in the FED
  // initialize with -1 (overflow) to start with 0 in the loop
  uint32_t globalECONDIdx = static_cast<uint32_t>(-1);

  // parse SLink body (capture blocks)
  for (uint32_t captureblockIdx = 0; captureblockIdx < HGCalMappingModuleIndexer::maxCBperFED_ && ptr < trailer - 2;
       captureblockIdx++) {
    // check capture block header (64b)
    LogDebug("[HGCalUnpacker]") << "@" << std::setw(8) << std::distance(header, ptr) << ": 0x" << std::hex
                                << std::setfill('0') << std::setw(16) << *ptr << std::dec;
    auto cb_header = *ptr;
    LogDebug("[HGCalUnpacker]") << "fedId = " << fedId << ", captureblockIdx = " << captureblockIdx
                                << ", cb_header = " << std::hex << std::setfill('0') << std::setw(16) << cb_header
                                << std::dec;
    // sanity check
    if (((cb_header >> (BACKEND_FRAME::CAPTUREBLOCK_RESERVED_POS + 32)) & BACKEND_FRAME::CAPTUREBLOCK_RESERVED_MASK) !=
        fedConfig.cbHeaderMarker) {
      //if word is a 0x0 it probably means that it's a 64b padding word: check that we are ending
      //the s-link may have less capture blocks than the maxCBperFED_ so for now this is considered normal
      uint32_t ECONDdenseIdx = moduleIndexer.getIndexForModule(fedId, 0);
      econdPacketInfo.view()[ECONDdenseIdx].location() = (uint32_t)(ptr - header);
      if (cb_header == 0x0) {
        auto nToEnd = (fed_data.size() / 8 - 2) - std::distance(header, ptr);
        if (nToEnd == 1) {
          ptr++;
          LogDebug("[HGCalUnpacker]")
              << "fedId = " << fedId
              << ", 64b padding word caught before parsing all max capture blocks, captureblockIdx = "
              << captureblockIdx;
          econdPacketInfo.view()[ECONDdenseIdx].exception() = 7;
          return UNPACKER_STAT::Normal;
        }
      }
      econdPacketInfo.view()[ECONDdenseIdx].exception() = 2;
      edm::LogWarning("[HGCalUnpacker]") << "Expected a capture block header at word " << std::dec
                                         << (uint32_t)(ptr - header) << "/0x" << std::hex << (uint32_t)(ptr - header)
                                         << " (reserved word: 0x" << fedConfig.cbHeaderMarker << "), got 0x"
                                         << ((cb_header >> (BACKEND_FRAME::CAPTUREBLOCK_RESERVED_POS + 32)) &
                                             BACKEND_FRAME::CAPTUREBLOCK_RESERVED_MASK)
                                         << " from 0x" << cb_header << ".";
      return UNPACKER_STAT::WrongCaptureBlockHeader;
    }
    ++ptr;
    // parse Capture Block body (ECON-Ds)
    for (uint32_t econdIdx = 0; econdIdx < HGCalMappingModuleIndexer::maxECONDperCB_; econdIdx++) {
      auto econd_pkt_status = (cb_header >> (3 * econdIdx)) & 0b111;
      LogDebug("[HGCalUnpacker]") << "fedId = " << fedId << ", captureblockIdx = " << captureblockIdx
                                  << ", econdIdx = " << econdIdx << ", econd_pkt_status = " << econd_pkt_status;
      if (econd_pkt_status != backend::ECONDPacketStatus::InactiveECOND) {
        // always increment the global ECON-D index (unless inactive/unconnected)
        globalECONDIdx++;
      }

      bool pkt_exists =
          (econd_pkt_status == backend::ECONDPacketStatus::Normal) ||
          (econd_pkt_status == backend::ECONDPacketStatus::PayloadCRCError) ||
          (econd_pkt_status == backend::ECONDPacketStatus::EventIDMismatch) ||
          (fedConfig.mismatchPassthroughMode && econd_pkt_status == backend::ECONDPacketStatus::BCIDOrbitIDMismatch);
      if (!pkt_exists) {
        continue;
      }

      // ECON-D header (two 32b words)
      LogDebug("[HGCalUnpacker]") << "@" << std::setw(8) << std::distance(header, ptr) << ": 0x" << std::hex
                                  << std::setfill('0') << std::setw(16) << *ptr << std::dec;
      auto econd_headers = to_32b_words(ptr);
      uint32_t ECONDdenseIdx = moduleIndexer.getIndexForModule(fedId, globalECONDIdx);
      econdPacketInfo.view()[ECONDdenseIdx].location() = (uint32_t)(ptr - header);
      // sanity check
      if (((econd_headers[0] >> ECOND_FRAME::HEADER_POS) & ECOND_FRAME::HEADER_MASK) !=
          fedConfig.econds[globalECONDIdx].headerMarker) {
        econdPacketInfo.view()[ECONDdenseIdx].exception() = 3;
        edm::LogWarning("[HGCalUnpacker]")
            << "Expected a ECON-D header at word " << std::dec << (uint32_t)(ptr - header) << "/0x" << std::hex
            << (uint32_t)(ptr - header) << " (marker: 0x" << fedConfig.econds[globalECONDIdx].headerMarker
            << "), got 0x" << econd_headers[0] << ".";
        return UNPACKER_STAT::WrongECONDHeader;
      }
      ++ptr;

      econdPacketInfo.view()[ECONDdenseIdx].cbFlag() = (uint8_t)(econd_pkt_status);
      // ECON-D payload length (num of 32b words)
      // NOTE: in the capture blocks, ECON-D packets do not have the trailing IDLE word
      const auto econd_payload_length = ((econd_headers[0] >> ECOND_FRAME::PAYLOAD_POS) & ECOND_FRAME::PAYLOAD_MASK);
      if (econd_payload_length > 469) {
        econdPacketInfo.view()[ECONDdenseIdx].exception() = 4;
        edm::LogWarning("[HGCalUnpacker]")
            << "Unpacked payload length=" << econd_payload_length << " exceeds the maximal length=469";
        return UNPACKER_STAT::ECONDPayloadLengthOverflow;
      }
      const auto econdFlag = ((econd_headers[0] >> ECOND_FRAME::BITT_POS) & 0b1111111) +
                             (((econd_headers[1] >> ECOND_FRAME::BITS_POS) & 0b1) << hgcaldigi::ECONDFlag::BITS_POS);
      econdPacketInfo.view()[ECONDdenseIdx].payloadLength() = (uint16_t)econd_payload_length;
      econdPacketInfo.view()[ECONDdenseIdx].econdFlag() = (uint8_t)econdFlag;
      econdPacketInfo.view()[ECONDdenseIdx].exception() = 0;

      // convert ECON-D packets into 32b words -- need to swap the order of the two 32b words in the 64b word
      auto econd_payload = to_econd_payload(ptr, econd_payload_length);

      // forward ptr to the next ECON-D; use integer division with (... + 1) / 2 to round up
      ptr += (econd_payload_length + 1) / 2;

      LogDebug("[HGCalUnpacker]") << "fedId = " << fedId << ", captureblockIdx = " << captureblockIdx
                                  << ", econdIdx = " << econdIdx << ", econd_headers = " << std::hex
                                  << std::setfill('0') << std::setw(8) << econd_headers[0] << " " << econd_headers[1]
                                  << std::dec << ", econd_payload_length = " << econd_payload_length;

      //quality check for ECON-D (no need to check again econd_pkt_status here again)
      if ((((econd_headers[0] >> ECOND_FRAME::HT_POS) & ECOND_FRAME::HT_MASK) >= 0b10) ||
          (((econd_headers[0] >> ECOND_FRAME::EBO_POS) & ECOND_FRAME::EBO_MASK) >= 0b10) ||
          (((econd_headers[0] >> ECOND_FRAME::BITM_POS) & 0b1) == 0) ||
          (((econd_headers[0] >> ECOND_FRAME::BITM_POS) & 0b1) == 0) || econd_payload_length == 0 || headerOnlyMode) {
        continue;
      }

      // parse ECON-D body(eRx subpackets)
      const auto enabledErx = fedReadoutSequence.enabledErx_[globalECONDIdx];
      const auto erxMax = moduleIndexer.getGlobalTypesNErx()[fedReadoutSequence.readoutTypes_[globalECONDIdx]];
      const bool pass_through_mode = (econd_headers[0] >> ECOND_FRAME::BITP_POS) & 0b1;

      unsigned iword = 0;
      if (!pass_through_mode) {
        // Standard ECON-D
        LogDebug("[HGCalUnpacker]") << "Standard ECON-D, erxMax=" << erxMax << "enabledErx= " << enabledErx;
        for (uint32_t erxIdx = 0; erxIdx < erxMax; erxIdx++) {
          // check if the eRx is enabled
          if ((enabledErx >> erxIdx & 1) == 0) {
            continue;
          }
          LogDebug("[HGCalUnpacker]") << "fedId = " << fedId << ", captureblockIdx = " << captureblockIdx
                                      << ", econdIdx = " << econdIdx << ", erxIdx=" << erxIdx;

          econdPacketInfo.view()[ECONDdenseIdx].cm()(erxIdx, 0) =
              (econd_payload[iword] >> ECOND_FRAME::COMMONMODE0_POS) & ECOND_FRAME::COMMONMODE0_MASK;
          econdPacketInfo.view()[ECONDdenseIdx].cm()(erxIdx, 1) =
              (econd_payload[iword] >> ECOND_FRAME::COMMONMODE1_POS) & ECOND_FRAME::COMMONMODE1_MASK;
          // check if the eRx sub-packet is empty (the "F" flag in the eRx sub-packet header)
          if (((econd_payload[iword] >> ECOND_FRAME::ERXFORMAT_POS) & ECOND_FRAME::ERXFORMAT_MASK) == 1) {
            LogDebug("[HGCalUnpacker]") << "eRx " << erxIdx << " is empty";
            iword += 1;  // length of an empty eRx header (32 bits)
            continue;    // go to the next eRx
          }

          // erx header
          uint16_t cmSum = ((econd_payload[iword] >> ECOND_FRAME::COMMONMODE0_POS) & ECOND_FRAME::COMMONMODE0_MASK) +
                           ((econd_payload[iword] >> ECOND_FRAME::COMMONMODE1_POS) & ECOND_FRAME::COMMONMODE1_MASK);
          uint64_t erxHeader = ((uint64_t)econd_payload[iword] << 32) | ((uint64_t)econd_payload[iword + 1]);
          LogDebug("[HGCalUnpacker]") << "erx_headers = 0x" << std::hex << std::setfill('0') << std::setw(16)
                                      << erxHeader << ", cmSum = " << std::dec << cmSum;
          iword += 2;

          // parse erx body (channel data)
          uint32_t iBit = 0;
          for (uint32_t channelIdx = 0; channelIdx < HGCalMappingCellIndexer::maxChPerErx_; channelIdx++) {
            uint32_t denseIdx = moduleIndexer.getIndexForModuleData(fedId, globalECONDIdx, erxIdx, channelIdx);

            // check if the channel has data
            if (((erxHeader >> channelIdx) & 1) == 0) {
              continue;
            }

            const uint32_t tempIndex = iBit / 32 + iword;
            const uint32_t tempBit = iBit % 32;
            const uint32_t temp = (tempBit == 0) ? econd_payload[tempIndex]
                                                 : (econd_payload[tempIndex] << tempBit) |
                                                       (econd_payload[tempIndex + 1] >> (32 - tempBit));
            const uint32_t code = temp >> 28;
            digis.view()[denseIdx].tctp() = tctp_[code];
            digis.view()[denseIdx].adcm1() = (temp >> adcm1Shift_[code]) & adcm1Mask_[code];
            digis.view()[denseIdx].adc() = (temp >> adcShift_[code]) & adcMask_[code];
            digis.view()[denseIdx].tot() = (temp >> totShift_[code]) & totMask_[code];
            digis.view()[denseIdx].toa() = (temp >> toaShift_[code] & toaMask_[code]);
            digis.view()[denseIdx].cm() = cmSum;
            digis.view()[denseIdx].flags() = 0;
            iBit += erxBodyBits_[code];
          }
          iword += iBit / 32;
          if (iBit % 32 != 0) {
            iword += 1;
          }
        }
      } else {
        // Passthrough ECON-D
        LogDebug("[HGCalUnpacker]") << "Passthrough ECON-D, erxMax=" << erxMax << "enabledErx= " << enabledErx;
        for (uint32_t erxIdx = 0; erxIdx < erxMax; erxIdx++) {
          // check if the eRx is enabled
          if ((enabledErx >> erxIdx & 1) == 0) {
            continue;
          }
          LogDebug("[HGCalUnpacker]") << "fedId = " << fedId << ", captureblockIdx = " << captureblockIdx
                                      << ", econdIdx = " << econdIdx << ", erxIdx=" << erxIdx;

          econdPacketInfo.view()[ECONDdenseIdx].cm()(erxIdx, 0) =
              (econd_payload[iword] >> ECOND_FRAME::COMMONMODE0_POS) & ECOND_FRAME::COMMONMODE0_MASK;
          econdPacketInfo.view()[ECONDdenseIdx].cm()(erxIdx, 1) =
              (econd_payload[iword] >> ECOND_FRAME::COMMONMODE1_POS) & ECOND_FRAME::COMMONMODE1_MASK;
          // check if the eRx sub-packet is empty (the "F" flag in the eRx sub-packet header)
          if (((econd_payload[iword] >> ECOND_FRAME::ERXFORMAT_POS) & ECOND_FRAME::ERXFORMAT_MASK) == 1) {
            LogDebug("[HGCalUnpacker]") << "eRx " << erxIdx << " is empty";
            iword += 1;  // length of an empty eRx header (32 bits)
            continue;    // go to the next eRx
          }

          // erx header
          uint16_t cmSum = ((econd_payload[iword] >> ECOND_FRAME::COMMONMODE0_POS) & ECOND_FRAME::COMMONMODE0_MASK) +
                           ((econd_payload[iword] >> ECOND_FRAME::COMMONMODE1_POS) & ECOND_FRAME::COMMONMODE1_MASK);
          uint64_t erxHeader = ((uint64_t)econd_payload[iword] << 32) | ((uint64_t)econd_payload[iword + 1]);
          LogDebug("[HGCalUnpacker]") << "erx_headers = 0x" << std::hex << std::setfill('0') << std::setw(16)
                                      << erxHeader << ", cmSum = " << std::dec << cmSum;
          iword += 2;

          // parse erx body (channel data)
          for (uint32_t channelIdx = 0; channelIdx < HGCalMappingCellIndexer::maxChPerErx_; channelIdx++) {
            uint32_t denseIdx = moduleIndexer.getIndexForModuleData(fedId, globalECONDIdx, erxIdx, channelIdx);

            // check if the channel has data
            if (((erxHeader >> channelIdx) & 1) == 0) {
              continue;
            }

            // check if in characterization mode
            if (fedConfig.econds[globalECONDIdx].rocs[erxIdx / 2].charMode) {
              //characterization mode
              digis.view()[denseIdx].tctp() = (econd_payload[iword] >> 30) & 0b11;
              digis.view()[denseIdx].adcm1() = 0;
              digis.view()[denseIdx].adc() = (econd_payload[iword] >> 20) & 0b1111111111;
              digis.view()[denseIdx].tot() = (econd_payload[iword] >> 10) & 0b1111111111;
              digis.view()[denseIdx].toa() = econd_payload[iword] & 0b1111111111;
              digis.view()[denseIdx].cm() = cmSum;
              digis.view()[denseIdx].flags() = hgcal::DIGI_FLAG::Characterization;
            } else {
              //not characteristic mode
              digis.view()[denseIdx].tctp() = (econd_payload[iword] >> 30) & 0b11;

              digis.view()[denseIdx].adcm1() = (econd_payload[iword] >> 20) & 0b1111111111;
              if (econd_payload[iword] >> 31 & 0b1) {
                digis.view()[denseIdx].adc() = 0;
                digis.view()[denseIdx].tot() = (econd_payload[iword] >> 10) & 0b1111111111;
              } else {
                digis.view()[denseIdx].adc() = (econd_payload[iword] >> 10) & 0b1111111111;
                digis.view()[denseIdx].tot() = 0;
              }
              digis.view()[denseIdx].toa() = econd_payload[iword] & 0b1111111111;
              digis.view()[denseIdx].cm() = cmSum;
              digis.view()[denseIdx].flags() = hgcal::DIGI_FLAG::Normal;
            }
            iword += 1;
          }
        }
      }
      // end of ECON-D parsing
      if (iword != econd_payload_length - 1) {
        econdPacketInfo.view()[ECONDdenseIdx].exception() = 5;
        edm::LogWarning("[HGCalUnpacker]")
            << "Mismatch between unpacked and expected ECON-D #" << (int)globalECONDIdx << " payload length\n"
            << "  unpacked payload length=" << iword + 1 << "\n"
            << "  expected payload length=" << econd_payload_length;
        return UNPACKER_STAT::ECONDPayloadLengthMismatch;
      }
    }
  }
  // skip the padding word as the last capture block will be aligned to 128b if needed
  if (std::distance(ptr, header) % 2) {
    ++ptr;
  }
  // check SLink trailer (128b)
  // TODO
  if (ptr + 2 != trailer) {
    uint32_t ECONDdenseIdx = moduleIndexer.getIndexForModule(fedId, 0);
    econdPacketInfo.view()[ECONDdenseIdx].exception() = 6;
    edm::LogWarning("[HGCalUnpacker]") << "Error finding the S-link trailer, expected at" << std::dec
                                       << (uint32_t)(trailer - header) << "/0x" << std::hex
                                       << (uint32_t)(trailer - header) << "Unpacked trailer at" << std::dec
                                       << (uint32_t)(trailer - header + 2) << "/0x" << std::hex
                                       << (uint32_t)(ptr - header + 2);
    return UNPACKER_STAT::WrongSLinkTrailer;
  }
  return UNPACKER_STAT::Normal;
}
