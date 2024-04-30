#include "EventFilter/HGCalRawToDigi/interface/HGCalUnpacker.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <array>

using namespace hgcal;

void HGCalUnpacker::parseFEDData(unsigned fedId,
                                 const FEDRawData& fed_data,
                                 const HGCalMappingModuleIndexer& moduleIndexer, 
                                 hgcaldigi::HGCalDigiHost& digis,
                                 bool headerOnlyMode) {

  // Readout Sequence for FED
  const auto fedReadoutSequence=moduleIndexer.fedReadoutSequences_[fedId];

  // Endianness assumption
  // From 32-bit word(ECOND) to 64-bit word(capture block): little endianness
  // Others: big endianness
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
  // counter for counting the index of ECON-D in the FEDID
  // use overflow to start with 0
  uint32_t globalECONDIdx=-1; 
  // parse SLink body (capture blocks)
  for (uint16_t captureblockIdx = 0; captureblockIdx < HGCalMappingModuleIndexer::maxCBperFED_ && ptr < trailer - 2;
       captureblockIdx++) {
    // check capture block header (64b)
    std::cout << "@" << std::setw(8) << std::distance(header, ptr) << ": 0x" << std::hex << std::setfill('0')
              << std::setw(16) << *ptr << std::dec << std::endl;
    auto cb_header = *ptr;
    ++ptr;
    std::cout << "fedId = " << fedId << ", captureblockIdx = " << captureblockIdx << ", cb_header = " << std::hex
              << std::setfill('0') << std::setw(16) << cb_header << std::dec << std::endl;

    // parse Capture Block body (ECON-Ds)
    for (uint16_t econdIdx = 0; econdIdx < HGCalMappingModuleIndexer::maxECONDperCB_; econdIdx++) {
      auto econd_pkt_status = (cb_header >> (3 * econdIdx)) & 0b111;
      std::cout << "fedId = " << fedId << ", captureblockIdx = " << captureblockIdx << ", econdIdx = " << econdIdx
                << ", econd_pkt_status = " << econd_pkt_status << std::endl;
      if (!(econd_pkt_status == backend::ECONDPacketStatus::InactiveECOND)){
        globalECONDIdx++;
      }
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

      if(((econd_headers[0] >> BITT_POS) & 0b1) == 1 || headerOnlyMode){
        // if econd is truncated or in headeronly unpacking, go to next ECON-D
        // use integer division with (... + 1) / 2 to round up
        ptr += (econd_payload_length + 1) / 2;
        continue;
      } 
      // parse ECON-D body(eRx subpackets)
      // convert ECON-D packets into 32b words -- need to swap the order of the two 32b words in the 64b word
      auto econd_payload = to_econd_payload(ptr, econd_payload_length);
      const auto enabledERX = fedReadoutSequence.enabledERX_[globalECONDIdx];
      const auto erxMax = moduleIndexer.globalTypesNErx_[fedReadoutSequence.readoutTypes_[globalECONDIdx]];
      uint16_t iPayload=2;//starting of ECON-D body
      if(((econd_headers[0]>>ECOND_FRAME::BITP_POS) & 0b1) == 0){
        //Standard ECON-D
        std::cout << "Standard ECON-D, erxMax=" << erxMax << "enabledErx= "<< enabledERX << std::endl; 
        for(uint8_t erxIdx = 0; erxIdx < erxMax; erxIdx++){
          // disabled check
          if ((enabledERX >> erxIdx & 1) == 0)
            continue;
          std::cout << "fedId = " << fedId << ", captureblockIdx = " << captureblockIdx << ", econdIdx = " << econdIdx << ", erxIdx=" << erxIdx <<std::endl;
          // empty check
          // currently ignoring empty eRx
          if (((econd_payload[iPayload] >> ECOND_FRAME::ERXFORMAT_POS) & ECOND_FRAME::ERXFORMAT_MASK) == 1) {
            std::cout << "eRx empty" << std::endl;
            iPayload += 1;  // length of an empty eRx header (32 bits)
            continue;    // go to the next eRx
          }
          // erx header
          uint16_t cm = ((econd_payload[iPayload] >> ECOND_FRAME::COMMONMODE0_POS) & ECOND_FRAME::COMMONMODE0_MASK)
                      + ((econd_payload[iPayload+1] >> ECOND_FRAME::COMMONMODE1_POS) & ECOND_FRAME::COMMONMODE1_MASK);
          uint64_t erxHeader = ((uint64_t)econd_payload[iPayload] << 32) | ((uint64_t)econd_payload[iPayload + 1]);
          uint32_t denseIdx = moduleIndexer.getIndexForModuleErx(fedId, globalECONDIdx, erxIdx);

          std::cout<< ", erx_headers = 0x" << std::hex << std::setfill('0') << std::setw(16) << erxHeader 
                   << ", cm = " << std::dec << cm << std::endl;
          iPayload+=2;
          // parse erx body (channel data)
          uint16_t iBit=0;
          for(uint32_t channelIdx=0; channelIdx < HGCalMappingCellIndexer::maxChPerErx_; channelIdx++){
              if (((erxHeader >> channelIdx) & 1) == 0)
                continue;  // only pick active channels
              denseIdx = denseIdx * HGCalMappingCellIndexer::maxChPerErx_ + channelIdx;
              const uint32_t tempIndex = iBit / 32 + iPayload;
              const uint8_t tempBit = iBit % 32;
              const uint32_t temp =
                  (tempBit == 0) ? econd_payload[tempIndex]
                                 : (econd_payload[tempIndex] << tempBit) | (econd_payload[tempIndex + 1] >> (32 - tempBit));
              const uint8_t code = temp >> 28;
              digis.view()[denseIdx].tctp() = tctp_[code];
              digis.view()[denseIdx].adcm1() = (temp >> adcm1Shift_[code]) & adcm1Mask_[code];
              digis.view()[denseIdx].adc() = (temp >> adcShift_[code]) & adcMask_[code];
              digis.view()[denseIdx].tot() = (temp >> totShift_[code]) & totMask_[code];
              digis.view()[denseIdx].toa() = (temp >> toaShift_[code] & toaMask_[code]);
              digis.view()[denseIdx].cm() = cm;
              digis.view()[denseIdx].flags() = 0;
              iBit += erxBodyBits_[code];
          }
          iPayload += iBit / 32;
          if (iBit % 32 != 0)
            iBit += 1;
        }
      }
      else{
        //Passthrough ECON-D
        std::cout << "Passthrough ECON-D, erxMax=" << erxMax << "enabledErx= "<< enabledERX << std::endl; 
        for(uint8_t erxIdx = 0; erxIdx < erxMax; erxIdx++){
          // disabled check
          if ((enabledERX >> erxIdx & 1) == 0)
            continue;
          std::cout << "fedId = " << fedId << ", captureblockIdx = " << captureblockIdx << ", econdIdx = " << econdIdx << ", erxIdx=" << erxIdx <<std::endl;
          // empty check
          // currently ignoring empty eRx
          if (((econd_payload[iPayload] >> ECOND_FRAME::ERXFORMAT_POS) & ECOND_FRAME::ERXFORMAT_MASK) == 1) {
            std::cout << "eRx empty" << std::endl;
            iPayload += 1;  // length of an empty eRx header (32 bits)
            continue;    // go to the next eRx
          }
          //erx header
          uint16_t cm = ((econd_payload[iPayload] >> ECOND_FRAME::COMMONMODE0_POS) & ECOND_FRAME::COMMONMODE0_MASK)
                      + ((econd_payload[iPayload+1] >> ECOND_FRAME::COMMONMODE1_POS) & ECOND_FRAME::COMMONMODE1_MASK);
          uint64_t erxHeader = ((uint64_t)econd_payload[iPayload] << 32) | ((uint64_t)econd_payload[iPayload + 1]);
          uint32_t denseIdx = moduleIndexer.getIndexForModuleErx(fedId, globalECONDIdx, erxIdx);

          std::cout<< ", erx_headers = 0x" << std::hex << std::setfill('0') << std::setw(16) << erxHeader 
                   << ", cm = " << std::dec << cm << std::endl;
          iPayload+=2;
          // parse erx body (channel data)
          for(uint32_t channelIdx=0; channelIdx < HGCalMappingCellIndexer::maxChPerErx_; channelIdx++){
            if (((erxHeader >> channelIdx) & 1) == 0)
              continue;  // only pick active channels
            denseIdx = denseIdx * HGCalMappingCellIndexer::maxChPerErx_ + channelIdx;
            //TODO: Check what to do with pass through ECOND
            digis.view()[denseIdx].tctp() = (econd_payload[iPayload] >> 30) & 0b11;
            digis.view()[denseIdx].adcm1() = 0;
            digis.view()[denseIdx].adc() = (econd_payload[iPayload] >> 20) & 0b1111111111;
            digis.view()[denseIdx].tot() = (econd_payload[iPayload] >> 10) & 0b1111111111;
            digis.view()[denseIdx].toa() = econd_payload[iPayload] & 0b1111111111;
            digis.view()[denseIdx].cm() = cm;
            digis.view()[denseIdx].flags() = 0;
            iPayload+=1;
          }
      }
      // ECON-D trailer
      // Nothing to process

      // forward ptr to the next ECON-D; use integer division with (... + 1) / 2 to round up
      ptr += (econd_payload_length + 1) / 2;
    }
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
