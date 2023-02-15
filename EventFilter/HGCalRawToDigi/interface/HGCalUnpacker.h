/*
Authors:
Yulun Miao, Northwestern University
Huilin Qu, CERN

Last Modified:

Description:
This class is designed to unpack raw data from HGCal, formatted as S-Links, Capture blocks, and ECONDs, to HGCROC channel data.

Functions in this class are

parse<raw data format>(uint32_t* inputArray, uint32_t inputSize, uint16_t (*enabledERXMapping)(uint16_t sLink, uint8_t captureBlock, uint8_t econd),D (*logicalMapping)(HGCalElectronicsId elecID)):
parse input in corresponding raw data format. Threes function are written, with reusable code pieces not integrated in functions, to improve performance
inputArray: input as 32-bits words.
inputSize: size of input array.
(*enabledERXMapping(uint16_t sLink, uint8_t captureBlock, uint8_t econd): map from slink, capture block, econd indices to enabled erx in this econd
(*logicalMapping)(HGCalElectronicsId elecID): logical mapping from HGCalElectronicsId to class D as ID

getChannelData(): return vector of HGCROCChannelDataFrame<D>(ID,value)
getCommonModeData(): return vector of 16-bit common mode data, lowest 10 bits is the ADC of the common mode, padding to 4 for half ROC turned on
getCommonModeIndex(): return vector of 32-bit index, the length is the same as getChannelData(), link from channel data to the first common mode on ROC (+0,+1,+2,+3 for all four common modes)
*/

#ifndef EventFilter_HGCalRawToDigi_HGCalUnpacker_h
#define EventFilter_HGCalRawToDigi_HGCalUnpacker_h

#include "DataFormats/HGCalDigi/interface/HGCROCChannelDataFrame.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"

#include <cstdint>
#include <vector>

struct HGCalUnpackerConfig {
  uint32_t sLinkBOE = 0x00;              //SLink BOE pattern
  uint32_t captureBlockReserved = 0x3F;  //capture block reserved pattern
  uint32_t econdHeaderMarker = 0x154;    //ECOND header Marker patter
  uint32_t sLinkCaptureBlockMax = 10;    //maximum number of capture blocks in one S-Link, default to be 10
  uint32_t captureBlockECONDMax = 12;    //maximum number of ECOND's in one capture block, default to be 12
  uint32_t econdERXMax = 12;             //maximum number of erx's in one ECOND, default to be 12
  uint32_t erxChannelMax = 37;           //maximum number of channels in one erx, default to be 37
  uint32_t payloadLengthMax = 469;       //maximum length of payload length
  uint32_t channelMax = 7000000;         //maximum number of channels unpacked
  uint32_t commonModeMax = 4000000;      //maximum number of common modes unpacked
  uint32_t badECONDMax = 200;            //maximum number of bad econds
};

template <class D>
class HGCalUnpacker {
public:
  enum SLinkHeaderShift {
    kSLinkBOEShift = 24,
  };
  enum SLinkHeaderMask {
    kSLinkBOEMask = 0b11111111,
  };
  enum CaptureBlockHeaderShift {
    kCaptureBlockReservedShift = 26,
  };
  enum CaptureBlockMask {
    kCaptureBlockReservedMask = 0b111111,
    kCaptureBlockECONDStatusMask = 0b111,
  };
  enum ECONDHeaderShift {
    kHeaderShift = 23,
    kPayloadLengthShift = 14,
    kPassThroughShfit = 13,
    kHTShift = 10,
    kEBOShift = 8,
    kMatchShift = 7,
    kTruncatedShift = 6,
  };
  enum ECONDHeaderMask {
    kHeaderMask = 0b111111111,
    kPayloadLengthMask = 0b111111111,
    kPassThroughMask = 0b1,
    kHTMask = 0b11,
    kEBOMask = 0b11,
    kMatchMask = 0b1,
    kTruncatedMask = 0b1,
  };
  enum ERXHeaderShift {
    kFormatShift = 25,
    kCommonmode0Shift = 15,
    kCommonmode1Shift = 5,
  };
  enum ERXHeaderMask {
    kFormatMask = 0b1,
    kCommonmode0Mask = 0b1111111111,
    kCommonmode1Mask = 0b1111111111,
  };

  HGCalUnpacker(HGCalUnpackerConfig config);

  void parseSLink(uint32_t* inputArray,
                  uint32_t inputSize,
                  uint16_t (*enabledERXMapping)(uint16_t sLink, uint8_t captureBlock, uint8_t econd),
                  D (*logicalMapping)(HGCalElectronicsId elecID));
  void parseCaptureBlock(uint32_t* inputArray,
                         uint32_t inputSize,
                         uint16_t (*enabledERXMapping)(uint16_t sLink, uint8_t captureBlock, uint8_t econd),
                         D (*logicalMapping)(HGCalElectronicsId elecID));
  void parseECOND(uint32_t* inputArray,
                  uint32_t inputSize,
                  uint16_t (*enabledERXMapping)(uint16_t sLink, uint8_t captureBlock, uint8_t econd),
                  D (*logicalMapping)(HGCalElectronicsId elecID));
  std::vector<HGCROCChannelDataFrame<D> > getChannelData();
  std::vector<uint32_t> getCommonModeIndex();
  std::vector<uint16_t> getCommonModeData();
  std::vector<uint32_t> getBadECOND();

private:
  const uint32_t erxBodyLeftShift_[16] = {2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const uint32_t erxBodyRightShift_[16] = {0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const uint32_t erxBodyMask_[16] = {0b00111111111111111111110000000000,
                                     0b00000000000011111111110000000000,
                                     0b00111111111111111111110000000000,
                                     0b00000000000011111111111111111111,
                                     0b00111111111111111111111111111111,
                                     0b00111111111111111111111111111111,
                                     0b00111111111111111111111111111111,
                                     0b00111111111111111111111111111111,
                                     0b11111111111111111111111111111111,
                                     0b11111111111111111111111111111111,
                                     0b11111111111111111111111111111111,
                                     0b11111111111111111111111111111111,
                                     0b11111111111111111111111111111111,
                                     0b11111111111111111111111111111111,
                                     0b11111111111111111111111111111111,
                                     0b11111111111111111111111111111111};
  const uint32_t erxBodyBits_[16] = {24, 16, 24, 24, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32};
  HGCalUnpackerConfig config_;
  HGCROCChannelDataFrame<D>* channelData_;  //Array for unpacked channels
  uint32_t* commonModeIndex_;               //Array for logicalMapping between unpacked channels to first common mode
  uint32_t channelDataSize_;                //Size of unpacked channels
  uint16_t* commonModeData_;                //Array for unpacked common modes
  uint32_t commonModeDataSize_;             //Size of unpacked common modes
  uint32_t* badECOND_;                      //Array of indices of bad ECOND
  uint32_t badECONDSize_;                   //Size of bad ECOND
};

#endif
