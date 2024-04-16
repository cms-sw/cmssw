/****************************************************************************
 *
 * Unpacker to decode HGCal raw data from SLinks / Capture blocks / ECON-Ds.
 * Authors:
 *   Yulun Miao, Northwestern University
 *   Huilin Qu, CERN
 *   Laurent Forthomme, CERN
 *
 ****************************************************************************/

#ifndef EventFilter_HGCalRawToDigi_HGCalUnpacker_h
#define EventFilter_HGCalRawToDigi_HGCalUnpacker_h

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"

// TODO @hqucms
// #include "DataFormats/HGCalDigi/interface/HGCalFlaggedECONDInfo.h"

#include <cstdint>
#include <functional>
#include <vector>

struct HGCalECONDFlags {
  // TODO @hqucms
  // revisit the content and move to a central place ?

  // uint32_t sLinkBOE{0x2a};            ///< S-Link BOE pattern
  // uint32_t cbHeaderMarker{0x5f};      ///< Capture block reserved pattern for a new event
  // uint32_t econdHeaderMarker{0x154};  ///< ECON-D header Marker pattern for a new event
  // uint32_t maxFEDsPerEndcap{512};     ///< maximum number of FEDs on one side
  // uint32_t sLinkCaptureBlockMax{10};  ///< maximum number of capture blocks in one S-Link
  // uint32_t captureBlockECONDMax{12};  ///< maximum number of ECON-Ds in one capture block
  // uint32_t econdERXMax{12};           ///< maximum number of eRxs in one ECON-D
  // uint32_t erxChannelMax{37};         ///< maximum number of channels in one eRx
  // uint32_t payloadLengthMax{469};     ///< maximum length of payload length
  // uint32_t channelMax{7000000};       ///< maximum number of channels unpacked
  // uint32_t commonModeMax{4000000};    ///< maximum number of common modes unpacked
};

struct HGCalFlaggedECONDInfo {
  // TODO @hqucms
  // implement this properly in a file in DataFormats/HGCalDigi
};

class HGCalUnpacker {
public:
  // TODO @hqucms
  // use central definitions in DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h

  // enum SLinkHeaderShift {
  //   kSLinkBOEShift = 24,
  //   kSLinkFEDIdShift = 0,
  // };
  // enum SLinkHeaderMask {
  //   kSLinkBOEMask = 0b11111111,
  //   kSLinkFEDIdMask = 0b1111111111,
  // };
  // enum CaptureBlockHeaderShift {
  //   kCaptureBlockReservedShift = 25,
  // };
  // enum CaptureBlockMask {
  //   kCaptureBlockReservedMask = 0b1111111,
  //   kCaptureBlockECONDStatusMask = 0b111,
  // };
  // enum ECONDHeaderShift {
  //   kHeaderShift = 23,
  //   kPayloadLengthShift = 14,
  //   kPassThroughShift = 13,
  //   kHTShift = 10,
  //   kEBOShift = 8,
  //   kMatchShift = 7,
  //   kTruncatedShift = 6,
  // };
  // enum ECONDHeaderMask {
  //   kHeaderMask = 0b111111111,
  //   kPayloadLengthMask = 0b111111111,
  //   kPassThroughMask = 0b1,
  //   kHTMask = 0b11,
  //   kEBOMask = 0b11,
  //   kMatchMask = 0b1,
  //   kTruncatedMask = 0b1,
  // };
  // enum ERXHeaderShift {
  //   kFormatShift = 25,
  //   kCommonmode0Shift = 15,
  //   kCommonmode1Shift = 5,
  // };
  // enum ERXHeaderMask {
  //   kFormatMask = 0b1,
  //   kCommonmode0Mask = 0b1111111111,
  //   kCommonmode1Mask = 0b1111111111,
  // };

  HGCalUnpacker() {}

  // TODO @hqucms
  // define what is needed as `config`
  // HGCalUnpacker(HGCalUnpackerConfig config);

  void parseFEDData(unsigned fedId,
                    const FEDRawData& fed_data,
                    hgcaldigi::HGCalDigiHost& digis,
                    hgcaldigi::HGCalDigiHost& common_modes,
                    std::vector<HGCalFlaggedECONDInfo>& errors);

private:
  // TODO @hqucms

  // const uint32_t erxBodyLeftShift_[16] = {2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // const uint32_t erxBodyRightShift_[16] = {0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // const uint32_t erxBodyMask_[16] = {0b00111111111111111111110000000000,
  //                                    0b00000000000011111111110000000000,
  //                                    0b00111111111111111111110000000000,
  //                                    0b00000000000011111111111111111111,
  //                                    0b00111111111111111111111111111111,
  //                                    0b00111111111111111111111111111111,
  //                                    0b00111111111111111111111111111111,
  //                                    0b00111111111111111111111111111111,
  //                                    0b11111111111111111111111111111111,
  //                                    0b11111111111111111111111111111111,
  //                                    0b11111111111111111111111111111111,
  //                                    0b11111111111111111111111111111111,
  //                                    0b11111111111111111111111111111111,
  //                                    0b11111111111111111111111111111111,
  //                                    0b11111111111111111111111111111111,
  //                                    0b11111111111111111111111111111111};
  // const uint32_t erxBodyBits_[16] = {24, 16, 24, 24, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32};
  // HGCalUnpackerConfig config_;
  // size_t channelDataSize_{0};                                             ///< Size of unpacked channels
  // size_t commonModeDataSize_{0};                                          ///< Size of unpacked common modes
  // std::vector<HGCROCChannelDataFrame<HGCalElectronicsId> > channelData_;  ///< Array for unpacked channels
  // std::vector<uint16_t> commonModeSum_;
  // std::vector<HGCROCChannelDataFrame<HGCalElectronicsId> > commonModeData_;  ///< Array for unpacked common modes
  // HGCalFlaggedECONDInfoCollection flaggedECOND_;                             ///< Array with flagged ECON-D information
};

#endif
