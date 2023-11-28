/****************************************************************************
 *
 * This is a part of HGCAL offline software.
 * Authors:
 *   Yulun Miao, Northwestern University
 *   Huilin Qu, CERN
 *   Laurent Forthomme, CERN
 *
 ****************************************************************************/

#include "EventFilter/HGCalRawToDigi/interface/HGCalUnpacker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

template <class D>
HGCalUnpacker<D>::HGCalUnpacker(HGCalUnpackerConfig config)
    : config_(config),
      channelData_(config_.channelMax),
      commonModeIndex_(config_.channelMax),
      commonModeData_(config_.commonModeMax) {}

template <class D>
void HGCalUnpacker<D>::parseSLink(
    const std::vector<uint32_t>& inputArray,
    const std::function<uint16_t(uint16_t sLink, uint8_t captureBlock, uint8_t econd)>& enabledERXMapping,
    const std::function<D(HGCalElectronicsId elecID)>& logicalMapping) {
  uint16_t sLink = 0;

  channelDataSize_ = 0;
  commonModeDataSize_ = 0;
  badECOND_.clear();

  for (uint32_t iword = 0; iword < inputArray.size();) {  // loop through the S-Link
    //----- parse the S-Link header
    if (((inputArray[iword] >> kSLinkBOEShift) & kSLinkBOEMask) != config_.sLinkBOE)  // sanity check
      throw cms::Exception("CorruptData")
          << "Expected a S-Link header at word " << std::dec << iword << "/0x" << std::hex << iword << " (BOE: 0x"
          << config_.sLinkBOE << "), got 0x" << inputArray[iword] << ".";

    iword += 4;  // length of the S-Link header (128 bits)

    LogDebug("HGCalUnpack") << "SLink=" << sLink;

    //----- parse the S-Link body
    for (uint8_t captureBlock = 0; captureBlock < config_.sLinkCaptureBlockMax;
         captureBlock++) {  // loop through all capture blocks
      //----- parse the capture block header
      if (((inputArray[iword] >> kCaptureBlockReservedShift) & kCaptureBlockReservedMask) !=
          config_.captureBlockReserved)  // sanity check
        throw cms::Exception("CorruptData")
            << "Expected a capture block header at word " << std::dec << iword << "/0x" << std::hex << iword
            << " (reserved word: 0x" << config_.captureBlockReserved << "), got 0x" << inputArray[iword] << ".";

      const uint64_t captureBlockHeader = ((uint64_t)inputArray[iword] << 32) | ((uint64_t)inputArray[iword + 1]);
      iword += 2;  // length of capture block header (64 bits)

      LogDebug("HGCalUnpack") << "Capture block=" << (int)captureBlock << ", capture block header=0x" << std::hex
                              << captureBlockHeader;

      //----- parse the capture block body
      for (uint8_t econd = 0; econd < config_.captureBlockECONDMax; econd++) {  // loop through all ECON-Ds
        if (((captureBlockHeader >> (3 * econd)) & kCaptureBlockECONDStatusMask) >= 0b100)
          continue;  // only pick active ECON-Ds

        //----- parse the ECON-D header
        // (the second word of ECON-D header contains no information for unpacking, use only the first one)
        if (((inputArray[iword] >> kHeaderShift) & kHeaderMask) != config_.econdHeaderMarker)  // sanity check
          throw cms::Exception("CorruptData")
              << "Expected a ECON-D header at word " << std::dec << iword << "/0x" << std::hex << iword
              << " (marker: 0x" << config_.econdHeaderMarker << "), got 0x" << inputArray[iword] << ".";

        const auto& econdHeader = inputArray[iword];
        iword += 2;  // length of ECON-D header (2 * 32 bits)

        LogDebug("HGCalUnpack") << "ECON-D #" << (int)econd << ", first word of ECON-D header=0x" << std::hex
                                << econdHeader;

        //----- extract the payload length
        const uint32_t payloadLength = (econdHeader >> kPayloadLengthShift) & kPayloadLengthMask;
        if (payloadLength > config_.payloadLengthMax)  // if payload length too big
          throw cms::Exception("CorruptData") << "Unpacked payload length=" << payloadLength
                                              << " exceeds the maximal length=" << config_.payloadLengthMax;

        LogDebug("HGCalUnpack") << "ECON-D #" << (int)econd << ", payload length=" << payloadLength;
        //Quality check
        if ((((captureBlockHeader >> (3 * econd)) & kCaptureBlockECONDStatusMask) != 0b000) ||
            (((econdHeader >> kHTShift) & kHTMask) >= 0b10) || (((econdHeader >> kEBOShift) & kEBOMask) >= 0b10) ||
            (((econdHeader >> kMatchShift) & kMatchMask) == 0) ||
            (((econdHeader >> kTruncatedShift) & kTruncatedMask) == 1)) {  // bad ECOND
          LogDebug("HGCalUnpack") << "ECON-D failed quality check, HT=" << (econdHeader >> kHTShift & kHTMask)
                                  << ", EBO=" << (econdHeader >> kEBOShift & kEBOMask)
                                  << ", M=" << (econdHeader >> kMatchShift & kMatchMask)
                                  << ", T=" << (econdHeader >> kTruncatedShift & kTruncatedMask);
          badECOND_.emplace_back(iword - 2);
          iword += payloadLength;  // skip the current ECON-D (using the payload length parsed above)

          if (iword % 2 != 0) {  //TODO: check this
            LogDebug("HGCalUnpacker") << "Padding ECON-D payload to 2 32-bit words (remainder: " << (iword % 2) << ").";
            iword += 1;
          }
          continue;  // go to the next ECON-D
        }
        const uint32_t econdBodyStart = iword;  // for the ECON-D length check
        //----- parse the ECON-D body
        if (((econdHeader >> kPassThroughShift) & kPassThroughMask) == 0) {
          // standard ECON-D
          LogDebug("HGCalUnpack") << "Standard ECON-D";
          const auto enabledERX = enabledERXMapping(sLink, captureBlock, econd);
          for (uint8_t erx = 0; erx < config_.econdERXMax; erx++) {
            //loop through eRx
            //pick active eRx
            if ((enabledERX >> erx & 1) == 0)
              continue;

            //----- parse the eRX subpacket header
            //common mode
            LogDebug("HGCalUnpack") << "ECON-D:eRx=" << (int)econd << ":" << (int)erx
                                    << ", first word of the eRx header=0x" << std::hex << inputArray[iword] << "\n"
                                    << "  extracted common mode 0=0x" << std::hex
                                    << ((inputArray[iword] >> kCommonmode0Shift) & kCommonmode0Mask) << std::dec
                                    << ", saved at " << commonModeDataSize_ << "\n"
                                    << "  extracted common mode 1=0x" << std::hex
                                    << ((inputArray[iword] >> kCommonmode1Shift) & kCommonmode1Mask) << std::dec
                                    << ", saved at " << (commonModeDataSize_ + 1);
            commonModeData_[commonModeDataSize_] = (inputArray[iword] >> kCommonmode0Shift) & kCommonmode0Mask;
            commonModeData_[commonModeDataSize_ + 1] = (inputArray[iword] >> kCommonmode1Shift) & kCommonmode1Mask;
            if ((erx % 2 == 0 && (enabledERX >> (erx + 1) & 1) == 0) ||
                (erx % 2 == 1 && (enabledERX >> (erx - 1) & 1) == 0)) {
              commonModeDataSize_ += 2;
              commonModeData_[commonModeDataSize_] = commonModeData_[commonModeDataSize_ - 2];
              commonModeData_[commonModeDataSize_ + 1] = commonModeData_[commonModeDataSize_ - 1];
              LogDebug("HGCalUnpack") << "half ROC turned on, padding to 4 common modes\n"
                                      << "0x" << std::hex << commonModeData_[commonModeDataSize_ - 2] << std::dec
                                      << " saved at " << commonModeDataSize_ << "\n"
                                      << "0x" << std::hex << commonModeData_[commonModeDataSize_ - 1] << std::dec
                                      << " saved at " << commonModeDataSize_ + 1;
            }
            // empty check
            if (((inputArray[iword] >> kFormatShift) & kFormatMask) == 1) {  // empty
              LogDebug("HGCalUnpack") << "eRx empty";
              iword += 1;  // length of an empty eRx header (32 bits)
              continue;    // go to the next eRx
            }
            // regular mode
            const uint64_t erxHeader = ((uint64_t)inputArray[iword] << 32) | ((uint64_t)inputArray[iword + 1]);
            iword += 2;  // length of a standard eRx header (2 * 32 bits)
            LogDebug("HGCalUnpack") << "whole eRx header=0x" << std::hex << erxHeader;

            //----- parse the eRx subpacket body
            uint32_t bitCounter = 0;
            for (uint8_t channel = 0; channel < config_.erxChannelMax; channel++) {  // loop through all channels in eRx
              if (((erxHeader >> channel) & 1) == 0)
                continue;  // only pick active channels
              const HGCalElectronicsId id(sLink, captureBlock, econd, erx, channel);
              commonModeIndex_[channelDataSize_] = commonModeDataSize_ / 4 * 4;
              const uint32_t tempIndex = bitCounter / 32 + iword;
              const uint8_t tempBit = bitCounter % 32;
              const uint32_t temp =
                  (tempBit == 0) ? inputArray[tempIndex]
                                 : (inputArray[tempIndex] << tempBit) | (inputArray[tempIndex + 1] >> (32 - tempBit));
              const uint8_t code = temp >> 28;
              // use if and else here
              channelData_[channelDataSize_] = HGCROCChannelDataFrame<D>(
                  logicalMapping(id),
                  ((temp << erxBodyLeftShift_[code]) >> erxBodyRightShift_[code]) & erxBodyMask_[code]);
              bitCounter += erxBodyBits_[code];
              if (code == 0b0010)
                channelData_[channelDataSize_].fillFlag1(1);
              LogDebug("HGCalUnpack") << "Word " << channelDataSize_ << ", ECON-D:eRx:channel=" << (int)econd << ":"
                                      << (int)erx << ":" << (int)channel << "\n"
                                      << "  assigned common mode index=" << commonModeIndex_.at(channelDataSize_)
                                      << "\n"
                                      << "  full word readout=0x" << std::hex << temp << std::dec << ", code=0x"
                                      << std::hex << (int)code << std::dec << "\n"
                                      << "  extracted channel data=0x" << std::hex
                                      << channelData_[channelDataSize_].raw();
              channelDataSize_++;
            }
            // pad to the whole word
            iword += bitCounter / 32;
            if (bitCounter % 32 != 0)
              iword += 1;

            if (commonModeDataSize_ + 1 > config_.commonModeMax)
              throw cms::Exception("HGCalUnpack") << "Too many common mode data unpacked: " << (commonModeDataSize_ + 1)
                                                  << " >= " << config_.commonModeMax << ".";
            commonModeDataSize_ += 2;
            // eRx subpacket has no trailer
          }
        } else {
          // passthrough ECON-D
          LogDebug("HGCalUnpack") << "Passthrough ECON-D";
          const auto enabledERX = enabledERXMapping(sLink, captureBlock, econd);
          for (uint8_t erx = 0; erx < config_.econdERXMax; erx++) {  // loop through all eRxs
            if ((enabledERX >> erx & 1) == 0)
              continue;  // only pick active eRxs

            //----- eRX subpacket header
            // common mode
            uint32_t temp = inputArray[iword];
            LogDebug("HGCalUnpack") << "ECON-D:eRx=" << (int)econd << ":" << (int)erx
                                    << ", first word of the eRx header=0x" << std::hex << temp << std::dec << "\n"
                                    << "  extracted common mode 0=0x" << std::hex
                                    << ((temp >> kCommonmode0Shift) & kCommonmode0Mask) << std::dec << ", saved at "
                                    << commonModeDataSize_ << "\n"
                                    << "  extracted common mode 1=0x" << std::hex
                                    << ((temp >> kCommonmode1Shift) & kCommonmode1Mask) << std::dec << ", saved at "
                                    << (commonModeDataSize_ + 1);
            commonModeData_[commonModeDataSize_] = (temp >> kCommonmode0Shift) & kCommonmode0Mask;
            commonModeData_[commonModeDataSize_ + 1] = (temp >> kCommonmode1Shift) & kCommonmode1Mask;
            if ((erx % 2 == 0 && (enabledERX >> (erx + 1) & 1) == 0) ||
                (erx % 2 == 1 && (enabledERX >> (erx - 1) & 1) == 0)) {
              commonModeDataSize_ += 2;
              commonModeData_[commonModeDataSize_] = commonModeData_[commonModeDataSize_ - 2];
              commonModeData_[commonModeDataSize_ + 1] = commonModeData_[commonModeDataSize_ - 1];
              LogDebug("HGCalUnpack") << "half ROC turned on, padding to 4 common modes\n"
                                      << "0x" << std::hex << commonModeData_[commonModeDataSize_ - 2] << std::dec
                                      << " saved at " << commonModeDataSize_ << "\n"
                                      << "0x" << std::hex << commonModeData_[commonModeDataSize_ - 1] << std::dec
                                      << " saved at " << commonModeDataSize_ + 1;
            }
            iword += 2;  // length of the standard eRx header (2 * 32 bits)
            for (uint8_t channel = 0; channel < config_.erxChannelMax; channel++) {  // loop through all channels in eRx
              const HGCalElectronicsId id(sLink, captureBlock, econd, erx, channel);
              commonModeIndex_[channelDataSize_] = commonModeDataSize_ / 4 * 4;
              channelData_[channelDataSize_] =
                  HGCROCChannelDataFrame<HGCalElectronicsId>(logicalMapping(id), inputArray[iword]);
              LogDebug("HGCalUnpack") << "Word " << channelDataSize_ << ", ECON-D:eRx:channel=" << (int)econd << ":"
                                      << (int)erx << ":" << (int)channel << ", HGCalElectronicsId=" << id.raw()
                                      << ", assigned common mode index=" << commonModeIndex_.at(channelDataSize_)
                                      << "\n"
                                      << "extracted channel data=0x" << std::hex
                                      << channelData_.at(channelDataSize_).raw();
              channelDataSize_++;
              iword++;
            }
            if (commonModeDataSize_ + 1 > config_.commonModeMax)
              throw cms::Exception("HGCalUnpack") << "Too many common mode data unpacked: " << (commonModeDataSize_ + 1)
                                                  << " >= " << config_.commonModeMax << ".";
            commonModeDataSize_ += 2;
          }
        }
        //----- parse the ECON-D trailer
        // (no information needed from ECON-D trailer in unpacker, skip it)
        iword += 1;  // length of the ECON-D trailer (32 bits CRC)

        if (iword - econdBodyStart != payloadLength)
          throw cms::Exception("CorruptData")
              << "Mismatch between unpacked and expected ECON-D #" << (int)econd << " payload length\n"
              << "  unpacked payload length=" << iword - econdBodyStart << "\n"
              << "  expected payload length=" << payloadLength;
        // pad to 2 words
        if (iword % 2 != 0) {  //TODO: check this
          LogDebug("HGCalUnpacker") << "Padding ECON-D payload to 2 32-bit words (remainder: " << (iword % 2) << ").";
          iword += 1;
        }
      }
      //----- capture block has no trailer
      // pad to 4 words
      if (iword % 4 != 0) {  //TODO: check this
        LogDebug("HGCalUnpacker") << "Padding capture block to 4 32-bit words (remainder: " << (iword % 4) << ").";
        iword += 4 - (iword % 4);
      }
    }
    //----- parse the S-Link trailer
    // (no information is needed in unpacker)

    iword += 4;  // length of the S-Link trailer (128 bits)
    sLink++;
  }
  channelData_.resize(channelDataSize_);
  commonModeIndex_.resize(channelDataSize_);
  commonModeData_.resize(commonModeDataSize_);
  return;
}

template <class D>
void HGCalUnpacker<D>::parseCaptureBlock(
    const std::vector<uint32_t>& inputArray,
    const std::function<uint16_t(uint16_t sLink, uint8_t captureBlock, uint8_t econd)>& enabledERXMapping,
    const std::function<D(HGCalElectronicsId elecID)>& logicalMapping) {
  uint16_t sLink = 0;
  uint8_t captureBlock = 0;

  channelDataSize_ = 0;
  commonModeDataSize_ = 0;
  badECOND_.clear();

  for (uint32_t iword = 0; iword < inputArray.size();) {  // loop through all capture blocks

    //----- parse the capture block header
    if (((inputArray[iword] >> kCaptureBlockReservedShift) & kCaptureBlockReservedMask) != config_.captureBlockReserved)
      throw cms::Exception("CorruptData")
          << "Expected a capture block header at word " << std::dec << iword << "/0x" << std::hex << iword
          << " (reserved word: 0x" << config_.captureBlockReserved << "), got 0x" << inputArray[iword] << ".";

    const uint64_t captureBlockHeader = ((uint64_t)inputArray[iword] << 32) | ((uint64_t)inputArray[iword + 1]);
    LogDebug("HGCalUnpack") << "Capture block=" << (int)captureBlock << ", capture block header=0x" << std::hex
                            << captureBlockHeader;
    iword += 2;  // length of capture block header (64 bits)

    //----- parse the capture block body
    for (uint8_t econd = 0; econd < config_.captureBlockECONDMax; econd++) {  // loop through all ECON-Ds
      if ((captureBlockHeader >> (3 * econd) & kCaptureBlockECONDStatusMask) >= 0b100)
        continue;  // only pick the active ECON-Ds

      //----- parse the ECON-D header
      // (the second word of ECON-D header contains no information useful for unpacking, use only the first one)
      if (((inputArray[iword] >> kHeaderShift) & kHeaderMask) != config_.econdHeaderMarker)  // sanity check
        throw cms::Exception("CorruptData")
            << "Expected a ECON-D header at word " << std::dec << iword << "/0x" << std::hex << iword << " (marker: 0x"
            << config_.econdHeaderMarker << "), got 0x" << inputArray[iword] << ".";

      const uint32_t econdHeader = inputArray[iword];
      iword += 2;  // length of ECON-D header (2 * 32 bits)

      LogDebug("HGCalUnpack") << "ECON-D #" << (int)econd << ", first word of ECON-D header=0x" << std::hex
                              << econdHeader;

      //----- extract the payload length
      const uint32_t payloadLength = ((econdHeader >> kPayloadLengthShift)) & kPayloadLengthMask;
      if (payloadLength > config_.payloadLengthMax)  // payload length is too large
        throw cms::Exception("CorruptData") << "Unpacked payload length=" << payloadLength
                                            << " exceeds the maximal length=" << config_.payloadLengthMax;
      LogDebug("HGCalUnpack") << "ECON-D #" << (int)econd << ", payload length = " << payloadLength;

      if ((((captureBlockHeader >> (3 * econd)) & kCaptureBlockECONDStatusMask) != 0b000) ||
          (((econdHeader >> kHTShift) & kHTMask) >= 0b10) || (((econdHeader >> kEBOShift) & kEBOMask) >= 0b10) ||
          (((econdHeader >> kMatchShift) & kMatchMask) == 0) ||
          (((econdHeader >> kTruncatedShift) & kTruncatedMask) == 1)) {  // quality check failed: bad ECON-D
        LogDebug("HGCalUnpack") << "ECON-D failed quality check, HT=" << (econdHeader >> kHTShift & kHTMask)
                                << ", EBO=" << (econdHeader >> kEBOShift & kEBOMask)
                                << ", M=" << (econdHeader >> kMatchShift & kMatchMask)
                                << ", T=" << (econdHeader >> kTruncatedShift & kTruncatedMask);
        badECOND_.emplace_back(iword - 2);
        iword += payloadLength;  // skip the current ECON-D (using the payload length parsed above)

        if (iword % 2 != 0) {  //TODO: check this
          LogDebug("HGCalUnpacker") << "Padding ECON-D payload to 2 32-bit words (remainder: " << (iword % 2) << ").";
          iword += 1;
        }
        continue;  // go to the next ECON-D
      }

      //----- parse the ECON-D body
      const uint32_t econdBodyStart = iword;  // for the ECON-D length check
      if (((econdHeader >> kPassThroughShift) & kPassThroughMask) == 0) {
        // standard ECON-D
        LogDebug("HGCalUnpack") << "Standard ECON-D";
        const auto enabledERX = enabledERXMapping(sLink, captureBlock, econd);
        for (uint8_t erx = 0; erx < config_.econdERXMax; erx++) {  // loop through all eRxs
          if ((enabledERX >> erx & 1) == 0)
            continue;  // only pick active eRx

          //----- parse the eRX subpacket header
          // common mode
          LogDebug("HGCalUnpack") << "ECON-D:eRx=" << (int)econd << ":" << (int)erx
                                  << ", first word of the eRx header=0x" << std::hex << inputArray[iword] << std::dec
                                  << "\n"
                                  << "  extracted common mode 0=0x" << std::hex
                                  << ((inputArray[iword] >> kCommonmode0Shift) & kCommonmode0Mask) << std::dec
                                  << ", saved at " << commonModeDataSize_ << "\n"
                                  << "  extracted common mode 1=0x" << std::hex
                                  << ((inputArray[iword] >> kCommonmode1Shift) & kCommonmode1Mask) << std::dec
                                  << ", saved at " << (commonModeDataSize_ + 1);

          commonModeData_[commonModeDataSize_] = (inputArray[iword] >> kCommonmode0Shift) & kCommonmode0Mask;
          commonModeData_[commonModeDataSize_ + 1] = (inputArray[iword] >> kCommonmode1Shift) & kCommonmode1Mask;
          if ((erx % 2 == 0 && (enabledERX >> (erx + 1) & 1) == 0) ||
              (erx % 2 == 1 && (enabledERX >> (erx - 1) & 1) == 0)) {
            commonModeDataSize_ += 2;
            commonModeData_[commonModeDataSize_] = commonModeData_[commonModeDataSize_ - 2];
            commonModeData_[commonModeDataSize_ + 1] = commonModeData_[commonModeDataSize_ - 1];
            LogDebug("HGCalUnpack") << "half ROC turned on, padding to 4 common modes\n"
                                    << "0x" << std::hex << commonModeData_[commonModeDataSize_ - 2] << std::dec
                                    << " saved at " << commonModeDataSize_ << "\n"
                                    << "0x" << std::hex << commonModeData_[commonModeDataSize_ - 1] << std::dec
                                    << " saved at " << commonModeDataSize_ + 1;
          }

          // empty check
          if (((inputArray[iword] >> kFormatShift) & kFormatMask) == 1) {
            iword += 1;  // length of an empty eRx header (32 bits)
            LogDebug("HGCalUnpack") << "eRx #" << (int)erx << " is empty.";
            continue;  // go to next eRx
          }

          // regular mode
          const uint64_t erxHeader = ((uint64_t)inputArray[iword] << 32) | (uint64_t)inputArray[iword + 1];
          LogDebug("HGCalUnpack") << "whole eRx header=0x" << std::hex << erxHeader;
          iword += 2;  // length of a standard eRx header (2 * 32 bits)

          uint32_t bitCounter = 0;
          //----- parse the eRx subpacket body
          for (uint8_t channel = 0; channel < config_.erxChannelMax; channel++) {  // loop through channels in eRx
            if (((erxHeader >> channel) & 1) == 0)
              continue;  // only pick active channels
            const HGCalElectronicsId id(sLink, captureBlock, econd, erx, channel);
            commonModeIndex_[channelDataSize_] = commonModeDataSize_ / 4 * 4;
            const uint32_t tempIndex = bitCounter / 32 + iword;
            const uint8_t tempBit = bitCounter % 32;
            const uint32_t temp =
                (tempBit == 0) ? inputArray[tempIndex]
                               : (inputArray[tempIndex] << tempBit) | (inputArray[tempIndex + 1] >> (32 - tempBit));
            const uint8_t code = temp >> 28;
            // use if and else here
            channelData_[channelDataSize_] = HGCROCChannelDataFrame<D>(
                logicalMapping(id),
                ((temp << erxBodyLeftShift_[code]) >> erxBodyRightShift_[code]) & erxBodyMask_[code]);
            bitCounter += erxBodyBits_[code];
            if (code == 0b0010)
              channelData_[channelDataSize_].fillFlag1(1);
            LogDebug("HGCalUnpack") << "Word " << channelDataSize_ << ", ECON-D:eRx:channel=" << (int)econd << ":"
                                    << (int)erx << ":" << (int)channel
                                    << ", assigned common mode index=" << commonModeIndex_[channelDataSize_] << "\n"
                                    << "  full word readout=0x" << std::hex << temp << std::dec << ", code=0x"
                                    << std::hex << (int)code << std::dec << "\n"
                                    << "  extracted channel data=0x" << std::hex
                                    << channelData_[channelDataSize_].raw();
            channelDataSize_++;
          }
          // pad to the whole word
          iword += bitCounter / 32;
          if (bitCounter % 32 != 0)
            iword += 1;

          if (commonModeDataSize_ + 1 > config_.commonModeMax)
            throw cms::Exception("HGCalUnpack") << "Too many common mode data unpacked: " << (commonModeDataSize_ + 1)
                                                << " >= " << config_.commonModeMax << ".";
          commonModeDataSize_ += 2;
          // eRx subpacket has no trailer
        }
      } else {  // passthrough ECON-D
        LogDebug("HGCalUnpack") << "Passthrough ECON-D";
        const auto enabledERX = enabledERXMapping(sLink, captureBlock, econd);
        for (uint8_t erx = 0; erx < config_.econdERXMax; erx++) {  // loop through all eRx
          if ((enabledERX >> erx & 1) == 0)
            continue;  // only pick active eRx

          //----- parse the eRX subpacket header
          // common mode
          uint32_t temp = inputArray[iword];
          LogDebug("HGCalUnpack") << "ECON-D:eRx=" << (int)econd << ":" << (int)erx
                                  << ", first word of the eRx header=0x" << std::hex << temp << std::dec << "\n"
                                  << "  extracted common mode 0=0x" << std::hex
                                  << ((temp >> kCommonmode0Shift) & kCommonmode0Mask) << std::dec << ", saved at "
                                  << commonModeDataSize_ << "\n"
                                  << "  extracted common mode 1=0x" << std::hex
                                  << ((temp >> kCommonmode1Shift) & kCommonmode1Mask) << std::dec << ", saved at "
                                  << (commonModeDataSize_ + 1);
          commonModeData_[commonModeDataSize_] = (temp >> kCommonmode0Shift) & kCommonmode0Mask;
          commonModeData_[commonModeDataSize_ + 1] = (temp >> kCommonmode1Shift) & kCommonmode1Mask;
          if ((erx % 2 == 0 && (enabledERX >> (erx + 1) & 1) == 0) ||
              (erx % 2 == 1 && (enabledERX >> (erx - 1) & 1) == 0)) {
            commonModeDataSize_ += 2;
            commonModeData_[commonModeDataSize_] = commonModeData_[commonModeDataSize_ - 2];
            commonModeData_[commonModeDataSize_ + 1] = commonModeData_[commonModeDataSize_ - 1];
            LogDebug("HGCalUnpack") << "half ROC turned on, padding to 4 common modes\n"
                                    << "0x" << std::hex << commonModeData_[commonModeDataSize_ - 2] << std::dec
                                    << " saved at " << commonModeDataSize_ << "\n"
                                    << "0x" << std::hex << commonModeData_[commonModeDataSize_ - 1] << std::dec
                                    << " saved at " << commonModeDataSize_ + 1;
          }
          iword += 2;  // length of a standard eRx header (2 * 32 bits)

          for (uint8_t channel = 0; channel < config_.erxChannelMax; channel++) {  // loop through all channels in eRx
            const HGCalElectronicsId id(sLink, captureBlock, econd, erx, channel);
            commonModeIndex_[channelDataSize_] = commonModeDataSize_ / 4 * 4;
            channelData_[channelDataSize_] =
                HGCROCChannelDataFrame<HGCalElectronicsId>(logicalMapping(id), inputArray[iword]);
            LogDebug("HGCalUnpack") << "Word" << channelDataSize_ << ", ECON-D:eRx:channel=" << (int)econd << ":"
                                    << (int)erx << ":" << (int)channel << ", HGCalElectronicsId=" << id.raw()
                                    << ", assigned common mode index=" << commonModeIndex_[channelDataSize_] << "\n"
                                    << "extracted channel data=0x" << std::hex << channelData_[channelDataSize_].raw();
            channelDataSize_++;
            iword++;
          }
          if (commonModeDataSize_ + 1 > config_.commonModeMax)
            throw cms::Exception("HGCalUnpack") << "Too many common mode data unpacked: " << (commonModeDataSize_ + 1)
                                                << " >= " << config_.commonModeMax << ".";
          commonModeDataSize_ += 2;
        }
      }

      //----- parse the ECON-D trailer
      // (no information unpacked from ECON-D trailer, just skip it)
      iword += 1;  // length of an ECON-D trailer (32 bits CRC)

      if (iword - econdBodyStart != payloadLength)
        throw cms::Exception("CorruptData")
            << "Mismatch between unpacked and expected ECON-D #" << (int)econd << " payload length\n"
            << "  unpacked payload length=" << iword - econdBodyStart << "\n"
            << "  expected payload length=" << payloadLength;
      // pad to 2 words
      if (iword % 2 != 0) {
        LogDebug("HGCalUnpacker") << "Padding ECON-D payload to 2 32-bit words (remainder: " << (iword % 2) << ").";
        iword += 1;
      }
    }
    captureBlock++;  // the capture block has no trailer to parse
  }
  channelData_.resize(channelDataSize_);
  commonModeIndex_.resize(channelDataSize_);
  commonModeData_.resize(commonModeDataSize_);
  return;
}

template <class D>
void HGCalUnpacker<D>::parseECOND(
    const std::vector<uint32_t>& inputArray,
    const std::function<uint16_t(uint16_t sLink, uint8_t captureBlock, uint8_t econd)>& enabledERXMapping,
    const std::function<D(HGCalElectronicsId elecID)>& logicalMapping) {
  uint16_t sLink = 0;
  uint8_t captureBlock = 0;
  uint8_t econd = 0;

  channelDataSize_ = 0;
  commonModeDataSize_ = 0;
  badECOND_.clear();

  for (uint32_t iword = 0; iword < inputArray.size();) {  // loop through all ECON-Ds
    //----- parse the ECON-D header
    // (the second word of ECON-D header contains no information for unpacking, use only the first one)
    if (((inputArray[iword] >> kHeaderShift) & kHeaderMask) != config_.econdHeaderMarker)  // sanity check
      throw cms::Exception("CorruptData")
          << "Expected a ECON-D header at word " << std::dec << iword << "/0x" << std::hex << iword << " (marker: 0x"
          << config_.econdHeaderMarker << "), got 0x" << inputArray[iword] << ".";

    const uint32_t econdHeader = inputArray[iword];
    iword += 2;  // length of ECON-D header (2 * 32 bits)

    LogDebug("HGCalUnpack") << "ECON-D #" << (int)econd << ", first word of ECON-D header=0x" << std::hex
                            << econdHeader;

    //----- extract the payload length
    const uint32_t payloadLength = (econdHeader >> kPayloadLengthShift) & kPayloadLengthMask;
    if (payloadLength > config_.payloadLengthMax)  // payload length too big
      throw cms::Exception("CorruptData")
          << "Unpacked payload length=" << payloadLength << " exceeds the maximal length=" << config_.payloadLengthMax;

    LogDebug("HGCalUnpack") << "ECON-D #" << (int)econd << ", payload length = " << payloadLength;
    //Quality check
    if (((econdHeader >> kHTShift & kHTMask) >= 0b10) || ((econdHeader >> kEBOShift & kEBOMask) >= 0b10) ||
        ((econdHeader >> kMatchShift & kMatchMask) == 0) ||
        ((econdHeader >> kTruncatedShift & kTruncatedMask) == 1)) {  // bad ECOND
      LogDebug("HGCalUnpack") << "ECON-D failed quality check, HT=" << (econdHeader >> kHTShift & kHTMask)
                              << ", EBO=" << (econdHeader >> kEBOShift & kEBOMask)
                              << ", M=" << (econdHeader >> kMatchShift & kMatchMask)
                              << ", T=" << (econdHeader >> kTruncatedShift & kTruncatedMask);
      badECOND_.emplace_back(iword - 2);
      iword += payloadLength;  // skip the current ECON-D (using the payload length parsed above)

      continue;  // go to the next ECON-D
    }

    //----- perse the ECON-D body
    const uint32_t econdBodyStart = iword;  // for the ECON-D length check
    if (((econdHeader >> kPassThroughShift) & kPassThroughMask) == 0) {
      // standard ECON-D
      LogDebug("HGCalUnpack") << "Standard ECON-D";
      const auto enabledERX = enabledERXMapping(sLink, captureBlock, econd);
      for (uint8_t erx = 0; erx < config_.econdERXMax; erx++) {  // loop through all eRxs
        if ((enabledERX >> erx & 1) == 0)
          continue;  // only pick active eRxs

        //----- parse the eRX subpacket header
        // common mode
        LogDebug("HGCalUnpack") << "ECON-D:eRx=" << (int)econd << ":" << (int)erx << ", first word of the eRx header=0x"
                                << std::hex << inputArray[iword] << std::dec << "\n"
                                << "  extracted common mode 0=0x" << std::hex
                                << ((inputArray[iword] >> kCommonmode0Shift) & kCommonmode0Mask) << std::dec
                                << ", saved at " << commonModeDataSize_ << "\n"
                                << "  extracted common mode 1=0x" << std::hex
                                << ((inputArray[iword] >> kCommonmode1Shift) & kCommonmode1Mask) << std::dec
                                << ", saved at " << (commonModeDataSize_ + 1);
        commonModeData_[commonModeDataSize_] = (inputArray[iword] >> kCommonmode0Shift) & kCommonmode0Mask;
        commonModeData_[commonModeDataSize_ + 1] = (inputArray[iword] >> kCommonmode1Shift) & kCommonmode1Mask;
        if ((erx % 2 == 0 && (enabledERX >> (erx + 1) & 1) == 0) ||
            (erx % 2 == 1 && (enabledERX >> (erx - 1) & 1) == 0)) {
          commonModeDataSize_ += 2;
          commonModeData_[commonModeDataSize_] = commonModeData_[commonModeDataSize_ - 2];
          commonModeData_[commonModeDataSize_ + 1] = commonModeData_[commonModeDataSize_ - 1];
          LogDebug("HGCalUnpack") << "half ROC turned on, padding to 4 common modes\n"
                                  << "0x" << std::hex << commonModeData_[commonModeDataSize_ - 2] << std::dec
                                  << " saved at " << commonModeDataSize_ << "\n"
                                  << "0x" << std::hex << commonModeData_[commonModeDataSize_ - 1] << std::dec
                                  << " saved at " << commonModeDataSize_ + 1;
        }
        if (((inputArray[iword] >> kFormatShift) & kFormatMask) == 1) {  // empty eRx
          LogDebug("HGCalUnpack") << "eRx empty";
          iword += 1;  // length of an empty eRx header (32 bits)

          continue;  // skip to the next eRx
        }

        // regular mode
        const uint64_t erxHeader = ((uint64_t)inputArray[iword] << 32) | ((uint64_t)inputArray[iword + 1]);
        iword += 2;  // length of a standard eRx header (2 * 32 bits)
        LogDebug("HGCalUnpack") << "whole eRx header=0x" << std::hex << erxHeader;

        //----- parse eRx subpacket body
        uint32_t bitCounter = 0;
        for (uint8_t channel = 0; channel < config_.erxChannelMax; channel++) {  // loop through all channels in eRx
          if (((erxHeader >> channel) & 1) == 0)
            continue;  // only pick active channels
          const HGCalElectronicsId id(sLink, captureBlock, econd, erx, channel);
          commonModeIndex_[channelDataSize_] = commonModeDataSize_ / 4 * 4;
          const uint32_t tempIndex = bitCounter / 32 + iword;
          const uint8_t tempBit = bitCounter % 32;
          const uint32_t temp =
              (tempBit == 0) ? inputArray[tempIndex]
                             : (inputArray[tempIndex] << tempBit) | (inputArray[tempIndex + 1] >> (32 - tempBit));
          const uint8_t code = temp >> 28;
          // use if and else here
          channelData_[channelDataSize_] = HGCROCChannelDataFrame<D>(
              logicalMapping(id), ((temp << erxBodyLeftShift_[code]) >> erxBodyRightShift_[code]) & erxBodyMask_[code]);
          bitCounter += erxBodyBits_[code];
          if (code == 0b0010)
            channelData_[channelDataSize_].fillFlag1(1);
          LogDebug("HGCalUnpack") << "Word " << channelDataSize_ << ", ECON-D:eRx:channel=" << (int)econd << ":"
                                  << (int)erx << ":" << (int)channel << "\n"
                                  << "  assigned common mode index=" << commonModeIndex_.at(channelDataSize_) << "\n"
                                  << "  full word readout=0x" << std::hex << temp << std::dec << ", code=0x" << std::hex
                                  << (int)code << std::dec << "\n"
                                  << "  extracted channel data=0x" << std::hex << channelData_[channelDataSize_].raw();
          channelDataSize_++;
        }
        // pad to the whole word
        iword += bitCounter / 32;
        if (bitCounter % 32 != 0)
          iword += 1;

        if (commonModeDataSize_ + 1 > config_.commonModeMax)
          throw cms::Exception("HGCalUnpack") << "Too many common mode data unpacked: " << (commonModeDataSize_ + 1)
                                              << " >= " << config_.commonModeMax << ".";
        commonModeDataSize_ += 2;
        // eRx subpacket has no trailer
      }
    } else {
      // passthrough ECON-D
      LogDebug("HGCalUnpack") << "Passthrough ECON-D";
      const auto enabledERX = enabledERXMapping(sLink, captureBlock, econd);
      for (uint8_t erx = 0; erx < config_.econdERXMax; erx++) {  // loop through all eRxs
        if ((enabledERX >> erx & 1) == 0)
          continue;  // only pick active eRx
        //----- parse the eRX subpacket header
        // common mode
        uint32_t temp = inputArray[iword];
        LogDebug("HGCalUnpack") << "ECON-D:eRx=" << (int)econd << ":" << (int)erx << ", first word of the eRx header=0x"
                                << std::hex << temp << std::dec << "\n"
                                << "  extracted common mode 0=0x" << std::hex
                                << ((temp >> kCommonmode0Shift) & kCommonmode0Mask) << std::dec << ", saved at "
                                << commonModeDataSize_ << "\n"
                                << "  extracted common mode 1=0x" << std::hex
                                << ((temp >> kCommonmode1Shift) & kCommonmode1Mask) << std::dec << ", saved at "
                                << (commonModeDataSize_ + 1);
        commonModeData_[commonModeDataSize_] = (temp >> kCommonmode0Shift) & kCommonmode0Mask;
        commonModeData_[commonModeDataSize_ + 1] = (temp >> kCommonmode1Shift) & kCommonmode1Mask;
        if ((erx % 2 == 0 && (enabledERX >> (erx + 1) & 1) == 0) ||
            (erx % 2 == 1 && (enabledERX >> (erx - 1) & 1) == 0)) {
          commonModeDataSize_ += 2;
          commonModeData_[commonModeDataSize_] = commonModeData_[commonModeDataSize_ - 2];
          commonModeData_[commonModeDataSize_ + 1] = commonModeData_[commonModeDataSize_ - 1];
          LogDebug("HGCalUnpack") << "half ROC turned on, padding to 4 common modes\n"
                                  << "0x" << std::hex << commonModeData_[commonModeDataSize_ - 2] << std::dec
                                  << " saved at " << commonModeDataSize_ << "\n"
                                  << "0x" << std::hex << commonModeData_[commonModeDataSize_ - 1] << std::dec
                                  << " saved at " << commonModeDataSize_ + 1;
        }
        iword += 2;  // length of the standard eRx header (2 * 32 bits)

        for (uint8_t channel = 0; channel < config_.erxChannelMax; channel++) {  // loop through all channels in eRx
          const HGCalElectronicsId id(sLink, captureBlock, econd, erx, channel);
          commonModeIndex_[channelDataSize_] = commonModeDataSize_ / 4 * 4;
          channelData_[channelDataSize_] =
              HGCROCChannelDataFrame<HGCalElectronicsId>(logicalMapping(id), inputArray[iword]);
          LogDebug("HGCalUnpack") << "Word " << channelDataSize_ << ", ECON-D:eRx:channel=" << (int)econd << ":"
                                  << (int)erx << ":" << (int)channel << ", HGCalElectronicsId=" << id.raw() << "\n"
                                  << "  assigned common mode index=" << commonModeIndex_.at(channelDataSize_) << "\n"
                                  << "extracted channel data=0x" << std::hex << channelData_.at(channelDataSize_).raw();
          channelDataSize_++;
          iword++;
        }
        if (commonModeDataSize_ + 1 > config_.commonModeMax)
          throw cms::Exception("HGCalUnpack") << "Too many common mode data unpacked: " << (commonModeDataSize_ + 1)
                                              << " >= " << config_.commonModeMax << ".";
        commonModeDataSize_ += 2;
      }
    }
    //----- fill the ECON-D trailer
    // (no information is needed from ECON-D trailer in unpacker, skip it)
    iword += 1;  // length of the ECON-D trailer (32 bits CRC)

    if (iword - econdBodyStart != payloadLength)
      throw cms::Exception("CorruptData")
          << "Mismatch between unpacked and expected ECON-D #" << (int)econd << " payload length\n"
          << "  unpacked payload length=" << iword - econdBodyStart << "\n"
          << "  expected payload length=" << payloadLength;
  }
  channelData_.resize(channelDataSize_);
  commonModeIndex_.resize(channelDataSize_);
  commonModeData_.resize(commonModeDataSize_);
  return;
}

// class specialisation for the electronics ID indexing
template class HGCalUnpacker<HGCalElectronicsId>;
