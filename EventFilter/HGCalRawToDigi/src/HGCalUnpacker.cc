#include "EventFilter/HGCalRawToDigi/interface/HGCalUnpacker.h"

#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <array>

using namespace hgcal;

void HGCalUnpacker::parseFEDData(unsigned fedId,
                                 const FEDRawData& fed_data,
                                 hgcaldigi::HGCalDigiHost& digis,
                                 hgcaldigi::HGCalDigiHost& common_modes,
                                 std::vector<HGCalFlaggedECONDInfo>& errors) {
  const auto* const header = reinterpret_cast<const uint64_t*>(fed_data.data());
  const auto* const trailer = reinterpret_cast<const uint64_t*>(fed_data.data() + fed_data.size());
  std::cout << "fedId = " << fedId << " nwords = " << std::distance(header, trailer) << std::endl;

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

  const auto* ptr = header;
  for (unsigned iword = 0; ptr < trailer; ++iword) {
    std::cout << std::setw(8) << iword << ": 0x" << std::hex << std::setfill('0') << std::setw(16) << *ptr << " ("
              << std::setfill('0') << std::setw(8) << *(reinterpret_cast<const uint32_t*>(ptr) + 1) << " "
              << std::setfill('0') << std::setw(8) << *reinterpret_cast<const uint32_t*>(ptr) << ")" << std::dec
              << std::endl;
    ++ptr;
  }
  std::cout << "@@@\n";
  ptr = header;

  // check SLink header (128b)
  // TODO
  ptr += 2;

  // parse capture blocks
  for (uint16_t captureblockIdx = 0; captureblockIdx < HGCalMappingModuleIndexer::maxCBperFED_ && ptr < trailer - 2;
       captureblockIdx++) {
    // check capture block header (64b)
    std::cout << "@" << std::setw(8) << std::distance(header, ptr) << ": 0x" << std::hex << std::setfill('0')
              << std::setw(16) << *ptr << std::dec << std::endl;
    auto cb_header = *ptr;
    ++ptr;
    std::cout << "fedId = " << fedId << ", captureblockIdx = " << captureblockIdx << ", cb_header = " << std::hex
              << std::setfill('0') << std::setw(16) << cb_header << std::dec << std::endl;

    // parse ECON-Ds
    for (uint16_t econdIdx = 0; econdIdx < HGCalMappingModuleIndexer::maxECONDperCB_; econdIdx++) {
      auto econd_pkt_status = (cb_header >> (3 * econdIdx)) & 0b111;
      std::cout << "fedId = " << fedId << ", captureblockIdx = " << captureblockIdx << ", econdIdx = " << econdIdx
                << ", econd_pkt_status = " << econd_pkt_status << std::endl;
      bool pkt_exists =
          (econd_pkt_status == backend::ECONDPacketStatus::Normal) ||
          (econd_pkt_status == backend::ECONDPacketStatus::PayloadCRCError) ||
          (econd_pkt_status == backend::ECONDPacketStatus::EventIDMismatch) ||
          (econd_pkt_status == backend::ECONDPacketStatus::BCIDOrbitIDMismatch);  // TODO: `BCIDOrbitIDMismatch`
      if (!pkt_exists) {
        continue;
      }

      // ECON-D header (two 32b words)
      std::cout << "@" << std::setw(8) << std::distance(header, ptr) << ": 0x" << std::hex << std::setfill('0')
                << std::setw(16) << *ptr << std::dec << std::endl;
      auto econd_headers = to_32b_words(ptr);
      ++ptr;

      // ECON-D payload length (num of 32b words)
      // NOTE: in the capture blocks, ECON-D packets do not have the trailing IDLE word
      const auto econd_payload_length = ((econd_headers[0] >> ECOND_FRAME::PAYLOAD_POS) & ECOND_FRAME::PAYLOAD_MASK);

      std::cout << "fedId = " << fedId << ", captureblockIdx = " << captureblockIdx << ", econdIdx = " << econdIdx
                << ", econd_headers = " << std::hex << std::setfill('0') << std::setw(8) << econd_headers[0] << " "
                << econd_headers[1] << std::dec << ", econd_payload_length = " << econd_payload_length << std::endl;

      // convert ECON-D packets into 32b words -- need to swap the order of the two 32b words in the 64b word
      auto econd_payload = to_econd_payload(ptr, econd_payload_length);

      // forward ptr to the next ECON-D; use integer division with (... + 1) / 2 to round up
      ptr += (econd_payload_length + 1) / 2;
    }

    // skip the padding word as capture blocks are padded to 128b
    if (std::distance(ptr, header) % 2) {
      ++ptr;
    }
  }

  // check SLink trailer (128b)
  // TODO
  assert(ptr + 2 == trailer);
}
