/*
Authors:
Yulun Miao, Northwestern University
Huilin Qu, CERN
*/

#include "EventFilter/HGCalRawToDigi/interface/HGCalUnpacker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

template <class D>
HGCalUnpacker<D>::HGCalUnpacker(HGCalUnpackerConfig config) : config_(config) {
  channelData_ =
      (HGCROCChannelDataFrame<D>*)malloc(config_.channelMax * sizeof(HGCROCChannelDataFrame<HGCalElectronicsId>));
  commonModeIndex_ = (uint32_t*)malloc(config_.channelMax * sizeof(uint32_t));
  commonModeData_ = (uint16_t*)malloc(config_.commonModeMax * sizeof(uint16_t));
  badECOND_ = (uint32_t*)malloc(config_.badECONDMax * sizeof(uint32_t));
}

template <class D>
void HGCalUnpacker<D>::parseSLink(uint32_t* inputArray,
                                  uint32_t inputSize,
                                  uint16_t (*enabledERXMapping)(uint16_t sLink, uint8_t captureBlock, uint8_t econd),
                                  D (*logicalMapping)(HGCalElectronicsId elecID)) {
  uint32_t temp;
  uint32_t payloadLength;
  uint16_t sLink = 0;
  uint8_t captureBlock = 0;
  uint8_t econd = 0;
  uint8_t erx = 0;
  uint8_t channel = 0;
  uint16_t enabledERX;

  uint64_t captureBlockHeader;
  uint32_t econdHeader;
  uint64_t erxHeader;
  uint32_t econdBodyStart;

  uint32_t bitCounter;
  uint32_t tempIndex;
  uint8_t tempBit;
  uint8_t code;

  HGCalElectronicsId id;
  channelDataSize_ = 0;
  commonModeDataSize_ = 0;
  badECONDSize_ = 0;

  uint32_t i = 0;
  while (i < inputSize) {
    //Loop through SLink
    //SLink header
    //Sanity check
    if (((inputArray[i] >> kSLinkBOEShift) & kSLinkBOEMask) == config_.sLinkBOE) {
      i += 4;  //Length of Slink header
    } else {
      //reading word that is not Slink header
      throw cms::Exception("CorruptData") << "Currently reading:" << std::hex << inputArray[i] << ", not S-Link header";
    }
    LogDebug("HGCalUnpack") << "SLink=" << sLink;

    //SLink body
    for (captureBlock = 0; captureBlock < config_.sLinkCaptureBlockMax; captureBlock++) {
      //Loop through Capture block
      //Capture block header
      //Sanity check
      if (((inputArray[i] >> kCaptureBlockReservedShift) & kCaptureBlockReservedMask) == config_.captureBlockReserved) {
        captureBlockHeader = ((uint64_t)inputArray[i] << 32) | ((uint64_t)inputArray[i + 1]);
        LogDebug("HGCalUnpack") << "Capture block=" << (int)captureBlock << " , capture block header=" << std::hex
                                << captureBlockHeader;
        i += 2;  //Length of capture block header
      } else {
        //reading word that is not capture block header
        throw cms::Exception("CorruptData")
            << "Currently reading:" << std::hex << inputArray[i] << ", not capture block header";
      }
      //Capture block body
      for (econd = 0; econd < config_.captureBlockECONDMax; econd++) {
        //Loop through ECON-D
        //Pick active ECON-D
        if (((captureBlockHeader >> (3 * econd)) & kCaptureBlockECONDStatusMask) >= 0b100)
          continue;
        //ECON-D header
        //The second word of ECON-D header contains no information for unpacking, use only the first one
        //Sanity check
        if (((inputArray[i] >> kHeaderShift) & kHeaderMask) == config_.econdHeaderMarker) {
          econdHeader = inputArray[i];
          LogDebug("HGCalUnpack") << "ECOND=" << (int)econd << " , first word of ECOND header=" << std::hex
                                  << econdHeader;
          i += 2;  //Length of ECON-D header
        } else {
          //if reading word that is not ECON-D header
          throw cms::Exception("CorruptData")
              << "Currently reading:" << std::hex << inputArray[i] << ", not ECOND header";
        }
        //Extract payloadLength
        payloadLength = (econdHeader >> kPayloadLengthShift) & kPayloadLengthMask;
        if (payloadLength > config_.payloadLengthMax) {
          //if payloadlength too big
          throw cms::Exception("CorruptData") << "payload length=" << payloadLength << ", too long";
        }
        LogDebug("HGCalUnpack") << "ECOND=" << (int)econd << ", payload length =" << payloadLength;
        //Quality check
        if ((((captureBlockHeader >> (3 * econd)) & kCaptureBlockECONDStatusMask) != 0b000) ||
            (((econdHeader >> kHTShift) & kHTMask) >= 0b10) || (((econdHeader >> kEBOShift) & kEBOMask) >= 0b10) ||
            (((econdHeader >> kMatchShift) & kMatchMask) == 0) ||
            (((econdHeader >> kTruncatedShift) & kTruncatedMask) == 1)) {
          //bad ECOND
          LogDebug("HGCalUnpack") << "ECOND failed quality check, HT=" << (econdHeader >> kHTShift & kHTMask)
                                  << ", EBO=" << (econdHeader >> kEBOShift & kEBOMask)
                                  << ", M=" << (econdHeader >> kMatchShift & kMatchMask)
                                  << ", T=" << (econdHeader >> kTruncatedShift & kTruncatedMask);
          badECOND_[badECONDSize_] = i - 2;
          badECONDSize_++;
          i += payloadLength;  //Skip current econd using payloadlength
          if (i % 2 != 0) {
            //TODO: check this
            i += 1;
          }
          continue;  //Go to next ECOND
        }
        econdBodyStart = i;  //For ECON-D length check
        //ECON-D body
        if (((econdHeader >> kPassThroughShfit) & kPassThroughMask) == 0) {
          //standard ECOND
          LogDebug("HGCalUnpack") << "Standard ECOND";
          enabledERX = enabledERXMapping(sLink, captureBlock, econd);
          for (erx = 0; erx < config_.econdERXMax; erx++) {
            //loop through eRx
            //pick active eRx
            if ((enabledERX >> erx & 1) == 0)
              continue;
            //eRX subpacket header
            //common mode
            LogDebug("HGCalUnpack") << "ECOND:erx=" << (int)econd << ":" << (int)erx
                                    << ", first word of the erx header=" << std::hex << inputArray[i];
            LogDebug("HGCalUnpack") << "Extract common mode 0=" << std::hex
                                    << ((inputArray[i] >> kCommonmode0Shift) & kCommonmode0Mask) << ", saved at "
                                    << commonModeDataSize_;
            LogDebug("HGCalUnpack") << "Extract common mode 1=" << std::hex
                                    << ((inputArray[i] >> kCommonmode1Shift) & kCommonmode1Mask) << ", saved at "
                                    << (commonModeDataSize_ + 1);
            commonModeData_[commonModeDataSize_] = (inputArray[i] >> kCommonmode0Shift) & kCommonmode0Mask;
            commonModeData_[commonModeDataSize_ + 1] = (inputArray[i] >> kCommonmode1Shift) & kCommonmode1Mask;
            if ((erx % 2 == 0 && (enabledERX >> (erx + 1) & 1) == 0) ||
                (erx % 2 == 1 && (enabledERX >> (erx - 1) & 1) == 0)) {
              LogDebug("HGCalUnpack") << "half ROC turned on, padding to 4 common modes";
              commonModeDataSize_ += 2;
              commonModeData_[commonModeDataSize_] = commonModeData_[commonModeDataSize_ - 2];
              commonModeData_[commonModeDataSize_ + 1] = commonModeData_[commonModeDataSize_ - 1];
              LogDebug("HGCalUnpack") << commonModeData_[commonModeDataSize_ - 2] << "saved at" << commonModeDataSize_;
              LogDebug("HGCalUnpack") << commonModeData_[commonModeDataSize_ - 1] << "saved at"
                                      << commonModeDataSize_ + 1;
            }
            //empty check
            if (((inputArray[i] >> kFormatShift) & kFormatMask) == 1) {
              LogDebug("HGCalUnpack") << "erx empty";
              //empty
              i += 1;    //Length of empty eRx header
              continue;  //Go to next eRx
            }
            //regular
            erxHeader = ((uint64_t)inputArray[i] << 32) | ((uint64_t)inputArray[i + 1]);
            i += 2;  //Length of standard eRx header
            LogDebug("HGCalUnpack") << "whole erx header=" << std::hex << erxHeader;
            bitCounter = 0;
            //eRx subpacket body
            for (channel = 0; channel < config_.erxChannelMax; channel++) {
              //Loop through channels in eRx
              //Pick active channels
              if (((erxHeader >> channel) & 1) == 0)
                continue;
              id = HGCalElectronicsId(sLink, captureBlock, econd, erx, channel);
              commonModeIndex_[channelDataSize_] = commonModeDataSize_ / 4 * 4;
              LogDebug("HGCalUnpack") << channelDataSize_ << ", ECOND:erx:channel=" << (int)econd << ":" << (int)erx
                                      << ":" << (int)channel
                                      << ", assigned commom mode index=" << commonModeIndex_[channelDataSize_];
              tempIndex = bitCounter / 32 + i;
              tempBit = bitCounter % 32;
              if (tempBit == 0) {
                temp = inputArray[tempIndex];
              } else {
                temp = (inputArray[tempIndex] << tempBit) | (inputArray[tempIndex + 1] >> (32 - tempBit));
              }
              code = temp >> 28;
              LogDebug("HGCalUnpack") << "full word readout=" << std::hex << temp;
              LogDebug("HGCalUnpack") << ", code=" << std::hex << (int)code;
              //use if and else here
              channelData_[channelDataSize_] = HGCROCChannelDataFrame<D>(
                  logicalMapping(id),
                  ((temp << erxBodyLeftShift_[code]) >> erxBodyRightShift_[code]) & erxBodyMask_[code]);
              bitCounter += erxBodyBits_[code];
              if (code == 0b0010) {
                channelData_[channelDataSize_].fillFlag1(1);
              }
              LogDebug("HGCalUnpack") << "extracted channel data=" << std::hex << channelData_[channelDataSize_].raw();
              channelDataSize_++;
            }
            //Pad to whole word
            i += bitCounter / 32;
            if (bitCounter % 32 != 0) {
              i += 1;
            }
            commonModeDataSize_ += 2;
            //eRx subpacket has no trailer
          }
        } else {
          //Pass through ECOND
          LogDebug("HGCalUnpack") << "Pass through ECOND";
          enabledERX = enabledERXMapping(sLink, captureBlock, econd);
          for (erx = 0; erx < config_.econdERXMax; erx++) {
            //loop through eRx
            //pick active eRx
            if ((enabledERX >> erx & 1) == 0)
              continue;
            //eRX subpacket header
            //common mode
            temp = inputArray[i];
            LogDebug("HGCalUnpack") << "ECOND:erx=" << (int)econd << ":" << (int)erx
                                    << ", first word of the erx header=" << std::hex << temp;
            LogDebug("HGCalUnpack") << "Extract common mode 0=" << ((temp >> kCommonmode0Shift) & kCommonmode0Mask)
                                    << ", saved at " << commonModeDataSize_;
            LogDebug("HGCalUnpack") << "Extract common mode 1=" << ((temp >> kCommonmode1Shift) & kCommonmode1Mask)
                                    << ", saved at " << (commonModeDataSize_ + 1);
            commonModeData_[commonModeDataSize_] = (temp >> kCommonmode0Shift) & kCommonmode0Mask;
            commonModeData_[commonModeDataSize_ + 1] = (temp >> kCommonmode1Shift) & kCommonmode1Mask;
            if ((erx % 2 == 0 && (enabledERX >> (erx + 1) & 1) == 0) ||
                (erx % 2 == 1 && (enabledERX >> (erx - 1) & 1) == 0)) {
              LogDebug("HGCalUnpack") << "half ROC turned on, padding to 4 common modes";
              commonModeDataSize_ += 2;
              commonModeData_[commonModeDataSize_] = commonModeData_[commonModeDataSize_ - 2];
              commonModeData_[commonModeDataSize_ + 1] = commonModeData_[commonModeDataSize_ - 1];
              LogDebug("HGCalUnpack") << commonModeData_[commonModeDataSize_ - 2] << "saved at" << commonModeDataSize_;
              LogDebug("HGCalUnpack") << commonModeData_[commonModeDataSize_ - 1] << "saved at"
                                      << commonModeDataSize_ + 1;
            }
            i += 2;  //Length of standard eRx header
            for (channel = 0; channel < config_.erxChannelMax; channel++) {
              //loop through channels in eRx
              id = HGCalElectronicsId(sLink, captureBlock, econd, erx, channel);
              commonModeIndex_[channelDataSize_] = commonModeDataSize_ / 4 * 4;
              LogDebug("HGCalUnpack") << channelDataSize_ << ", ECOND:erx:channel=" << (int)econd << ":" << (int)erx
                                      << ":" << (int)channel << ", HGCalElectronicsId=" << id.raw()
                                      << ", assigned commom mode index=" << commonModeIndex_[channelDataSize_];
              channelData_[channelDataSize_] =
                  HGCROCChannelDataFrame<HGCalElectronicsId>(logicalMapping(id), inputArray[i]);
              LogDebug("HGCalUnpack") << "extracted channel data=" << std::hex << channelData_[channelDataSize_].raw();
              channelDataSize_++;
              i++;
            }
            commonModeDataSize_ += 2;
          }
        }
        //ECON-D trailer
        //No information needed from ECON-D trailer in unpacker, skip it
        i += 1;  //Length of ECOND trailer
        //Check consisitency between length unpacked and payload length
        if ((i - econdBodyStart) != payloadLength) {
          throw cms::Exception("CorruptData")
              << "mismatch between length unpacked and payload length, length=" << i - econdBodyStart
              << ", payload length=" << payloadLength;
        }
        //Pad to 2 words
        if (i % 2 != 0) {
          //TODO: check this
          i += 1;
        }
      }
      //Capture block has no trailer
      //Pad to 4 words
      if (i % 4 != 0) {
        //TODO: check this
        i = i + 4 - (i % 4);
      }
    }
    //SLink trailer
    //No information needed in unpacker
    i += 4;  //Length of SLink trailer
    sLink++;
  }
  return;
}

template <class D>
void HGCalUnpacker<D>::parseCaptureBlock(uint32_t* inputArray,
                                         uint32_t inputSize,
                                         uint16_t (*enabledERXMapping)(uint16_t sLink,
                                                                       uint8_t captureBlock,
                                                                       uint8_t econd),
                                         D (*logicalMapping)(HGCalElectronicsId elecID)) {
  uint32_t temp;
  uint32_t payloadLength;
  uint16_t sLink = 0;
  uint8_t captureBlock = 0;
  uint8_t econd = 0;
  uint8_t erx = 0;
  uint8_t channel = 0;
  uint16_t enabledERX;

  uint64_t captureBlockHeader;
  uint32_t econdHeader;
  uint64_t erxHeader;
  uint32_t econdBodyStart;

  uint32_t bitCounter;
  uint32_t tempIndex;
  uint8_t tempBit;
  uint8_t code;

  HGCalElectronicsId id;
  channelDataSize_ = 0;
  commonModeDataSize_ = 0;
  badECONDSize_ = 0;

  uint32_t i = 0;
  while (i < inputSize) {
    //Loop through Capture block
    //Capture block header
    //Sanity check
    if (((inputArray[i] >> kCaptureBlockReservedShift) & kCaptureBlockReservedMask) == config_.captureBlockReserved) {
      captureBlockHeader = ((uint64_t)inputArray[i] << 32) | ((uint64_t)inputArray[i + 1]);
      LogDebug("HGCalUnpack") << "Capture block=" << (int)captureBlock << " , capture block header=" << std::hex
                              << captureBlockHeader;
      i += 2;  //Length of capture block header
    } else {
      //reading word that is not capture block header
      throw cms::Exception("CorruptData")
          << "Currently reading:" << std::hex << inputArray[i] << ", not capture block header";
    }
    //Capture block body
    for (econd = 0; econd < config_.captureBlockECONDMax; econd++) {
      //Loop through ECON-D
      //Pick active ECON-D
      if ((captureBlockHeader >> (3 * econd) & kCaptureBlockECONDStatusMask) >= 0b100)
        continue;
      //ECON-D header
      //The second word of ECON-D header contains no information for unpacking, use only the first one
      //Sanity check
      if (((inputArray[i] >> kHeaderShift) & kHeaderMask) == config_.econdHeaderMarker) {
        econdHeader = inputArray[i];
        LogDebug("HGCalUnpack") << "ECOND=" << (int)econd << " , first word of ECOND header=" << std::hex
                                << econdHeader;
        i += 2;  //Length of ECON-D header
      } else {
        //reading word that is not ECON-D header
        throw cms::Exception("CorruptData")
            << "Currently reading:" << std::hex << inputArray[i] << ", not ECOND header";
      }
      //Extract payloadLength
      payloadLength = ((econdHeader >> kPayloadLengthShift)) & kPayloadLengthMask;
      if (payloadLength > config_.payloadLengthMax) {
        //payloadlength too big
        throw cms::Exception("CorruptData") << "payload length=" << payloadLength << ", too long";
      }
      LogDebug("HGCalUnpack") << "ECOND=" << (int)econd << ", payload length = " << payloadLength;
      //Quality check
      if ((((captureBlockHeader >> (3 * econd)) & kCaptureBlockECONDStatusMask) != 0b000) ||
          (((econdHeader >> kHTShift) & kHTMask) >= 0b10) || (((econdHeader >> kEBOShift) & kEBOMask) >= 0b10) ||
          (((econdHeader >> kMatchShift) & kMatchMask) == 0) ||
          (((econdHeader >> kTruncatedShift) & kTruncatedMask) == 1)) {
        //bad ECOND
        LogDebug("HGCalUnpack") << "ECOND failed quality check, HT=" << (econdHeader >> kHTShift & kHTMask)
                                << ", EBO=" << (econdHeader >> kEBOShift & kEBOMask)
                                << ", M=" << (econdHeader >> kMatchShift & kMatchMask)
                                << ", T=" << (econdHeader >> kTruncatedShift & kTruncatedMask);
        badECOND_[badECONDSize_] = i - 2;
        badECONDSize_++;
        i += payloadLength;  //Skip current econd using payloadlength
        if (i % 2 != 0) {
          //TODO: check this
          i += 1;
        }
        continue;  //Go to next ECOND
      }
      econdBodyStart = i;  //For ECON-D length check
      //ECON-D body
      if (((econdHeader >> kPassThroughShfit) & kPassThroughMask) == 0) {
        //standard ECOND
        LogDebug("HGCalUnpack") << "Standard ECOND";
        enabledERX = enabledERXMapping(sLink, captureBlock, econd);
        for (erx = 0; erx < config_.econdERXMax; erx++) {
          //loop through eRx
          //pick active eRx
          if ((enabledERX >> erx & 1) == 0)
            continue;
          //eRX subpacket header
          //common mode
          LogDebug("HGCalUnpack") << "ECOND:erx=" << (int)econd << ":" << (int)erx
                                  << ", first word of the erx header=" << std::hex << inputArray[i];
          LogDebug("HGCalUnpack") << "Extract common mode 0=" << std::hex
                                  << ((inputArray[i] >> kCommonmode0Shift) & kCommonmode0Mask) << ", saved at "
                                  << commonModeDataSize_;
          LogDebug("HGCalUnpack") << "Extract common mode 1=" << std::hex
                                  << ((inputArray[i] >> kCommonmode1Shift) & kCommonmode1Mask) << ", saved at "
                                  << (commonModeDataSize_ + 1);
          commonModeData_[commonModeDataSize_] = (inputArray[i] >> kCommonmode0Shift) & kCommonmode0Mask;
          commonModeData_[commonModeDataSize_ + 1] = (inputArray[i] >> kCommonmode1Shift) & kCommonmode1Mask;
          if ((erx % 2 == 0 && (enabledERX >> (erx + 1) & 1) == 0) ||
              (erx % 2 == 1 && (enabledERX >> (erx - 1) & 1) == 0)) {
            LogDebug("HGCalUnpack") << "half ROC turned on, padding to 4 common modes";
            commonModeDataSize_ += 2;
            commonModeData_[commonModeDataSize_] = commonModeData_[commonModeDataSize_ - 2];
            commonModeData_[commonModeDataSize_ + 1] = commonModeData_[commonModeDataSize_ - 1];
            LogDebug("HGCalUnpack") << commonModeData_[commonModeDataSize_ - 2] << "saved at" << commonModeDataSize_;
            LogDebug("HGCalUnpack") << commonModeData_[commonModeDataSize_ - 1] << "saved at"
                                    << commonModeDataSize_ + 1;
          }
          //empty check
          if (((inputArray[i] >> kFormatShift) & kFormatMask) == 1) {
            LogDebug("HGCalUnpack") << "erx empty";
            //empty
            i += 1;    //Length of empty eRx header
            continue;  //Go to next eRx
          }
          //regular
          erxHeader = ((uint64_t)inputArray[i] << 32) | ((uint64_t)inputArray[i + 1]);
          i += 2;  //Length of standard eRx header
          LogDebug("HGCalUnpack") << "whole erx header=" << std::hex << erxHeader;
          bitCounter = 0;
          //eRx subpacket body
          for (channel = 0; channel < config_.erxChannelMax; channel++) {
            //Loop through channels in eRx
            //Pick active channels
            if (((erxHeader >> channel) & 1) == 0)
              continue;
            id = HGCalElectronicsId(sLink, captureBlock, econd, erx, channel);
            commonModeIndex_[channelDataSize_] = commonModeDataSize_ / 4 * 4;
            LogDebug("HGCalUnpack") << channelDataSize_ << ", ECOND:erx:channel=" << (int)econd << ":" << (int)erx
                                    << ":" << (int)channel
                                    << ", assigned commom mode index=" << commonModeIndex_[channelDataSize_];
            tempIndex = bitCounter / 32 + i;
            tempBit = bitCounter % 32;
            if (tempBit == 0) {
              temp = inputArray[tempIndex];
            } else {
              temp = (inputArray[tempIndex] << tempBit) | (inputArray[tempIndex + 1] >> (32 - tempBit));
            }
            code = temp >> 28;
            LogDebug("HGCalUnpack") << "full word readout=" << std::hex << temp;
            LogDebug("HGCalUnpack") << ", code=" << std::hex << (int)code;
            //use if and else here
            channelData_[channelDataSize_] = HGCROCChannelDataFrame<D>(
                logicalMapping(id),
                ((temp << erxBodyLeftShift_[code]) >> erxBodyRightShift_[code]) & erxBodyMask_[code]);
            bitCounter += erxBodyBits_[code];
            if (code == 0b0010) {
              channelData_[channelDataSize_].fillFlag1(1);
            }
            LogDebug("HGCalUnpack") << "extracted channel data=" << std::hex << channelData_[channelDataSize_].raw();
            channelDataSize_++;
          }
          //Pad to whole word
          i += bitCounter / 32;
          if (bitCounter % 32 != 0) {
            i += 1;
          }
          commonModeDataSize_ += 2;
          //eRx subpacket has no trailer
        }
      } else {
        //Pass through ECOND
        LogDebug("HGCalUnpack") << "Pass through ECOND";
        enabledERX = enabledERXMapping(sLink, captureBlock, econd);
        for (erx = 0; erx < config_.econdERXMax; erx++) {
          //loop through eRx
          //pick active eRx
          if ((enabledERX >> erx & 1) == 0)
            continue;
          //eRX subpacket header
          //common mode
          temp = inputArray[i];
          LogDebug("HGCalUnpack") << "ECOND:erx=" << (int)econd << ":" << (int)erx
                                  << ", first word of the erx header=" << std::hex << temp;
          LogDebug("HGCalUnpack") << "Extract common mode 0=" << ((temp >> kCommonmode0Shift) & kCommonmode0Mask)
                                  << ", saved at " << commonModeDataSize_;
          LogDebug("HGCalUnpack") << "Extract common mode 1=" << ((temp >> kCommonmode1Shift) & kCommonmode1Mask)
                                  << ", saved at " << (commonModeDataSize_ + 1);
          commonModeData_[commonModeDataSize_] = (temp >> kCommonmode0Shift) & kCommonmode0Mask;
          commonModeData_[commonModeDataSize_ + 1] = (temp >> kCommonmode1Shift) & kCommonmode1Mask;
          if ((erx % 2 == 0 && (enabledERX >> (erx + 1) & 1) == 0) ||
              (erx % 2 == 1 && (enabledERX >> (erx - 1) & 1) == 0)) {
            LogDebug("HGCalUnpack") << "half ROC turned on, padding to 4 common modes";
            commonModeDataSize_ += 2;
            commonModeData_[commonModeDataSize_] = commonModeData_[commonModeDataSize_ - 2];
            commonModeData_[commonModeDataSize_ + 1] = commonModeData_[commonModeDataSize_ - 1];
            LogDebug("HGCalUnpack") << commonModeData_[commonModeDataSize_ - 2] << "saved at" << commonModeDataSize_;
            LogDebug("HGCalUnpack") << commonModeData_[commonModeDataSize_ - 1] << "saved at"
                                    << commonModeDataSize_ + 1;
          }
          i += 2;  //Length of standard eRx header
          for (channel = 0; channel < config_.erxChannelMax; channel++) {
            //loop through channels in eRx
            id = HGCalElectronicsId(sLink, captureBlock, econd, erx, channel);
            commonModeIndex_[channelDataSize_] = commonModeDataSize_ / 4 * 4;
            LogDebug("HGCalUnpack") << channelDataSize_ << ", ECOND:erx:channel=" << (int)econd << ":" << (int)erx
                                    << ":" << (int)channel << ", HGCalElectronicsId=" << id.raw()
                                    << ", assigned commom mode index=" << commonModeIndex_[channelDataSize_];
            channelData_[channelDataSize_] =
                HGCROCChannelDataFrame<HGCalElectronicsId>(logicalMapping(id), inputArray[i]);
            LogDebug("HGCalUnpack") << "extracted channel data=" << std::hex << channelData_[channelDataSize_].raw();
            channelDataSize_++;
            i++;
          }
          commonModeDataSize_ += 2;
        }
      }
      //ECON-D trailer
      //No information needed from ECON-D trailer in unpacker, skip it
      i += 1;  //Length of ECOND trailer
      //Check consisitency between length unpacked and payload length
      if ((i - econdBodyStart) != payloadLength) {
        throw cms::Exception("CorruptData")
            << "mismatch between length unpacked and payload length, length=" << i - econdBodyStart
            << ", payload length=" << payloadLength;
      }
      //Pad to 2 words
      if (i % 2 != 0) {
        i += 1;
      }
    }
    //Capture block has no trailer
    captureBlock++;
  }
  return;
}

template <class D>
void HGCalUnpacker<D>::parseECOND(uint32_t* inputArray,
                                  uint32_t inputSize,
                                  uint16_t (*enabledERXMapping)(uint16_t sLink, uint8_t captureBlock, uint8_t econd),
                                  D (*logicalMapping)(HGCalElectronicsId elecID)) {
  uint32_t temp;
  uint32_t payloadLength;
  uint16_t sLink = 0;
  uint8_t captureBlock = 0;
  uint8_t econd = 0;
  uint8_t erx = 0;
  uint8_t channel = 0;
  uint16_t enabledERX;

  uint64_t captureBlockHeader;
  uint32_t econdHeader;
  uint64_t erxHeader;
  uint32_t econdBodyStart;

  uint32_t bitCounter;
  uint32_t tempIndex;
  uint8_t tempBit;
  uint8_t code;

  HGCalElectronicsId id;
  channelDataSize_ = 0;
  commonModeDataSize_ = 0;
  badECONDSize_ = 0;

  uint32_t i = 0;
  while (i < inputSize) {
    //Loop through ECON-D
    //ECON-D header
    //The second word of ECON-D header contains no information for unpacking, use only the first one
    //Sanity check
    if (((inputArray[i] >> kHeaderShift) & kHeaderMask) == config_.econdHeaderMarker) {
      econdHeader = inputArray[i];
      LogDebug("HGCalUnpack") << "ECOND=" << (int)econd << " , first word of ECOND header=" << std::hex << econdHeader;
      i += 2;  //Length of ECON-D header
    } else {
      //reading word that is not ECON-D header
      throw cms::Exception("CorruptData") << "Currently reading:" << std::hex << inputArray[i] << ", not ECOND header";
    }
    //Extract payloadLength
    payloadLength = (econdHeader >> kPayloadLengthShift) & kPayloadLengthMask;
    if (payloadLength > config_.payloadLengthMax) {
      //payloadlength too big
      throw cms::Exception("CorruptData") << "payload length=" << payloadLength << ", too long";
    }
    LogDebug("HGCalUnpack") << "ECOND=" << (int)econd << ", payload length = " << payloadLength;
    //Quality check
    if (((econdHeader >> kHTShift & kHTMask) >= 0b10) || ((econdHeader >> kEBOShift & kEBOMask) >= 0b10) ||
        ((econdHeader >> kMatchShift & kMatchMask) == 0) || ((econdHeader >> kTruncatedShift & kTruncatedMask) == 1)) {
      //bad ECOND
      LogDebug("HGCalUnpack") << "ECOND failed quality check, HT=" << (econdHeader >> kHTShift & kHTMask)
                              << ", EBO=" << (econdHeader >> kEBOShift & kEBOMask)
                              << ", M=" << (econdHeader >> kMatchShift & kMatchMask)
                              << ", T=" << (econdHeader >> kTruncatedShift & kTruncatedMask);
      badECOND_[badECONDSize_] = i - 2;
      badECONDSize_++;
      i += payloadLength;  //Skip current econd using payloadlength
      continue;            //Go to next ECOND
    }
    econdBodyStart = i;  //For ECON-D length check
    //ECON-D body
    if (((econdHeader >> kPassThroughShfit) & kPassThroughMask) == 0) {
      //standard ECOND
      LogDebug("HGCalUnpack") << "Standard ECOND";
      enabledERX = enabledERXMapping(sLink, captureBlock, econd);
      for (erx = 0; erx < config_.econdERXMax; erx++) {
        //loop through eRx
        //pick active eRx
        if ((enabledERX >> erx & 1) == 0)
          continue;
        //eRX subpacket header
        //common mode
        LogDebug("HGCalUnpack") << "ECOND:erx=" << (int)econd << ":" << (int)erx
                                << ", first word of the erx header=" << std::hex << inputArray[i];
        LogDebug("HGCalUnpack") << "Extract common mode 0=" << std::hex
                                << ((inputArray[i] >> kCommonmode0Shift) & kCommonmode0Mask) << ", saved at "
                                << commonModeDataSize_;
        LogDebug("HGCalUnpack") << "Extract common mode 1=" << std::hex
                                << ((inputArray[i] >> kCommonmode1Shift) & kCommonmode1Mask) << ", saved at "
                                << (commonModeDataSize_ + 1);
        commonModeData_[commonModeDataSize_] = (inputArray[i] >> kCommonmode0Shift) & kCommonmode0Mask;
        commonModeData_[commonModeDataSize_ + 1] = (inputArray[i] >> kCommonmode1Shift) & kCommonmode1Mask;
        if ((erx % 2 == 0 && (enabledERX >> (erx + 1) & 1) == 0) ||
            (erx % 2 == 1 && (enabledERX >> (erx - 1) & 1) == 0)) {
          LogDebug("HGCalUnpack") << "half ROC turned on, padding to 4 common modes";
          commonModeDataSize_ += 2;
          commonModeData_[commonModeDataSize_] = commonModeData_[commonModeDataSize_ - 2];
          commonModeData_[commonModeDataSize_ + 1] = commonModeData_[commonModeDataSize_ - 1];
          LogDebug("HGCalUnpack") << commonModeData_[commonModeDataSize_ - 2] << "saved at" << commonModeDataSize_;
          LogDebug("HGCalUnpack") << commonModeData_[commonModeDataSize_ - 1] << "saved at" << commonModeDataSize_ + 1;
        }
        //empty check
        if (((inputArray[i] >> kFormatShift) & kFormatMask) == 1) {
          LogDebug("HGCalUnpack") << "erx empty";
          //empty
          i += 1;    //Length of empty eRx header
          continue;  //Go to next eRx
        }
        //regular
        erxHeader = ((uint64_t)inputArray[i] << 32) | ((uint64_t)inputArray[i + 1]);
        i += 2;  //Length of standard eRx header
        LogDebug("HGCalUnpack") << "whole erx header=" << std::hex << erxHeader;
        bitCounter = 0;
        //eRx subpacket body
        for (channel = 0; channel < config_.erxChannelMax; channel++) {
          //Loop through channels in eRx
          //Pick active channels
          if (((erxHeader >> channel) & 1) == 0)
            continue;
          id = HGCalElectronicsId(sLink, captureBlock, econd, erx, channel);
          commonModeIndex_[channelDataSize_] = commonModeDataSize_ / 4 * 4;
          LogDebug("HGCalUnpack") << channelDataSize_ << ", ECOND:erx:channel=" << (int)econd << ":" << (int)erx << ":"
                                  << (int)channel
                                  << ", assigned commom mode index=" << commonModeIndex_[channelDataSize_];
          tempIndex = bitCounter / 32 + i;
          tempBit = bitCounter % 32;
          if (tempBit == 0) {
            temp = inputArray[tempIndex];
          } else {
            temp = (inputArray[tempIndex] << tempBit) | (inputArray[tempIndex + 1] >> (32 - tempBit));
          }
          code = temp >> 28;
          LogDebug("HGCalUnpack") << "full word readout=" << std::hex << temp;
          LogDebug("HGCalUnpack") << ", code=" << std::hex << (int)code;
          //use if and else here
          channelData_[channelDataSize_] = HGCROCChannelDataFrame<D>(
              logicalMapping(id), ((temp << erxBodyLeftShift_[code]) >> erxBodyRightShift_[code]) & erxBodyMask_[code]);
          bitCounter += erxBodyBits_[code];
          if (code == 0b0010) {
            channelData_[channelDataSize_].fillFlag1(1);
          }
          LogDebug("HGCalUnpack") << "extracted channel data=" << std::hex << channelData_[channelDataSize_].raw();
          channelDataSize_++;
        }
        //Pad to whole word
        i += bitCounter / 32;
        if (bitCounter % 32 != 0) {
          i += 1;
        }
        commonModeDataSize_ += 2;
        //eRx subpacket has no trailer
      }
    } else {
      //Pass through ECOND
      LogDebug("HGCalUnpack") << "Pass through ECOND";
      enabledERX = enabledERXMapping(sLink, captureBlock, econd);
      for (erx = 0; erx < config_.econdERXMax; erx++) {
        //loop through eRx
        //pick active eRx
        if ((enabledERX >> erx & 1) == 0)
          continue;
        //eRX subpacket header
        //common mode
        temp = inputArray[i];
        LogDebug("HGCalUnpack") << "ECOND:erx=" << (int)econd << ":" << (int)erx
                                << ", first word of the erx header=" << std::hex << temp;
        LogDebug("HGCalUnpack") << "Extract common mode 0=" << ((temp >> kCommonmode0Shift) & kCommonmode0Mask)
                                << ", saved at " << commonModeDataSize_;
        LogDebug("HGCalUnpack") << "Extract common mode 1=" << ((temp >> kCommonmode1Shift) & kCommonmode1Mask)
                                << ", saved at " << (commonModeDataSize_ + 1);
        commonModeData_[commonModeDataSize_] = (temp >> kCommonmode0Shift) & kCommonmode0Mask;
        commonModeData_[commonModeDataSize_ + 1] = (temp >> kCommonmode1Shift) & kCommonmode1Mask;
        if ((erx % 2 == 0 && (enabledERX >> (erx + 1) & 1) == 0) ||
            (erx % 2 == 1 && (enabledERX >> (erx - 1) & 1) == 0)) {
          LogDebug("HGCalUnpack") << "half ROC turned on, padding to 4 common modes";
          commonModeDataSize_ += 2;
          commonModeData_[commonModeDataSize_] = commonModeData_[commonModeDataSize_ - 2];
          commonModeData_[commonModeDataSize_ + 1] = commonModeData_[commonModeDataSize_ - 1];
          LogDebug("HGCalUnpack") << commonModeData_[commonModeDataSize_ - 2] << "saved at" << commonModeDataSize_;
          LogDebug("HGCalUnpack") << commonModeData_[commonModeDataSize_ - 1] << "saved at" << commonModeDataSize_ + 1;
        }
        i += 2;  //Length of standard eRx header
        for (channel = 0; channel < config_.erxChannelMax; channel++) {
          //loop through channels in eRx
          id = HGCalElectronicsId(sLink, captureBlock, econd, erx, channel);
          commonModeIndex_[channelDataSize_] = commonModeDataSize_ / 4 * 4;
          LogDebug("HGCalUnpack") << channelDataSize_ << ", ECOND:erx:channel=" << (int)econd << ":" << (int)erx << ":"
                                  << (int)channel << ", HGCalElectronicsId=" << id.raw()
                                  << ", assigned commom mode index=" << commonModeIndex_[channelDataSize_];
          channelData_[channelDataSize_] =
              HGCROCChannelDataFrame<HGCalElectronicsId>(logicalMapping(id), inputArray[i]);
          LogDebug("HGCalUnpack") << "extracted channel data=" << std::hex << channelData_[channelDataSize_].raw();
          channelDataSize_++;
          i++;
        }
        commonModeDataSize_ += 2;
      }
    }
    //ECON-D trailer
    //No information needed from ECON-D trailer in unpacker, skip it
    i += 1;  //Length of ECOND trailer
    //Check consisitency between length unpacked and payload length
    if ((i - econdBodyStart) != payloadLength) {
      throw cms::Exception("CorruptData")
          << "mismatch between length unpacked and payload length, length=" << i - econdBodyStart
          << ", payload length=" << payloadLength;
    }
  }
  return;
}

template <class D>
std::vector<HGCROCChannelDataFrame<D> > HGCalUnpacker<D>::getChannelData() {
  return std::vector<HGCROCChannelDataFrame<D> >(channelData_, channelData_ + channelDataSize_);
}

template <class D>
std::vector<uint32_t> HGCalUnpacker<D>::getCommonModeIndex() {
  return std::vector<uint32_t>(commonModeIndex_, commonModeIndex_ + channelDataSize_);
}

template <class D>
std::vector<uint16_t> HGCalUnpacker<D>::getCommonModeData() {
  return std::vector<uint16_t>(commonModeData_, commonModeData_ + commonModeDataSize_);
}

template <class D>
std::vector<uint32_t> HGCalUnpacker<D>::getBadECOND() {
  return std::vector<uint32_t>(badECOND_, badECOND_ + badECONDSize_);
}

template class HGCalUnpacker<HGCalElectronicsId>;
