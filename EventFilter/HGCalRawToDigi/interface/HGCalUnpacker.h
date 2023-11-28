/****************************************************************************
 *
 * This is a part of HGCAL offline software.
 * Authors:
 *   Yulun Miao, Northwestern University
 *   Huilin Qu, CERN
 *   Laurent Forthomme, CERN
 *
 ****************************************************************************/

#ifndef EventFilter_HGCalRawToDigi_HGCalUnpacker_h
#define EventFilter_HGCalRawToDigi_HGCalUnpacker_h

#include "DataFormats/HGCalDigi/interface/HGCROCChannelDataFrame.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"

#include <cstdint>
#include <functional>
#include <vector>

struct HGCalUnpackerConfig {
  uint32_t sLinkBOE{0x0};               ///< S-Link BOE pattern
  uint32_t captureBlockReserved{0x3f};  ///< Capture block reserved pattern
  uint32_t econdHeaderMarker{0x154};    ///< ECON-D header Marker pattern
  uint32_t sLinkCaptureBlockMax{10};    ///< maximum number of capture blocks in one S-Link
  uint32_t captureBlockECONDMax{12};    ///< maximum number of ECON-Ds in one capture block
  uint32_t econdERXMax{12};             ///< maximum number of eRxs in one ECON-D
  uint32_t erxChannelMax{37};           ///< maximum number of channels in one eRx
  uint32_t payloadLengthMax{469};       ///< maximum length of payload length
  uint32_t channelMax{7000000};         ///< maximum number of channels unpacked
  uint32_t commonModeMax{4000000};      ///< maximum number of common modes unpacked
};

/// This class is designed to unpack raw data from HGCal, formatted as S-Links, capture blocks, and ECON-Ds, to HGCROC channel data.
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
    kPassThroughShift = 13,
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

  explicit HGCalUnpacker(HGCalUnpackerConfig config);

  /// parse input in S-Link format
  /// \param[in] inputArray input as 32-bits words vector.
  /// \param[in] enabledERXMapping map from S-Link indices to enabled eRx in this ECON-D
  /// \param[in] logicalMapping logical mapping from HGCalElectronicsId to class D as ID
  void parseSLink(const std::vector<uint32_t>& inputArray,
                  const std::function<uint16_t(uint16_t sLink, uint8_t captureBlock, uint8_t econd)>& enabledERXMapping,
                  const std::function<D(HGCalElectronicsId elecID)>& logicalMapping);
  /// parse input in capture block format
  /// \param[in] inputArray input as 32-bits words vector.
  /// \param[in] enabledERXMapping map from capture block indices to enabled eRx in this ECON-D
  /// \param[in] logicalMapping logical mapping from HGCalElectronicsId to class D as ID
  void parseCaptureBlock(
      const std::vector<uint32_t>& inputArray,
      const std::function<uint16_t(uint16_t sLink, uint8_t captureBlock, uint8_t econd)>& enabledERXMapping,
      const std::function<D(HGCalElectronicsId elecID)>& logicalMapping);
  /// parse input in ECON-D format
  /// \param[in] inputArray input as 32-bits words vector.
  /// \param[in] enabledERXMapping map from ECON-D indices to enabled eRx in this ECON-D
  /// \param[in] logicalMapping logical mapping from HGCalElectronicsId to class D as ID
  void parseECOND(const std::vector<uint32_t>& inputArray,
                  const std::function<uint16_t(uint16_t sLink, uint8_t captureBlock, uint8_t econd)>& enabledERXMapping,
                  const std::function<D(HGCalElectronicsId elecID)>& logicalMapping);

  /// \return vector of HGCROCChannelDataFrame<D>(ID, value)
  const std::vector<HGCROCChannelDataFrame<D> >& channelData() const { return channelData_; }
  /// \return vector of 32-bit index, the length is the same as channelData(), link from channel data to the first common mode on ROC (+0,+1,+2,+3 for all four common modes)
  const std::vector<uint32_t>& commonModeIndex() const { return commonModeIndex_; }
  /// \return vector of 16-bit common mode data, lowest 10 bits is the ADC of the common mode, padding to 4 for half ROC turned on
  const std::vector<uint16_t>& commonModeData() const { return commonModeData_; }
  const std::vector<uint32_t>& badECOND() const { return badECOND_; }

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
  size_t channelDataSize_{0};                            ///< Size of unpacked channels
  size_t commonModeDataSize_{0};                         ///< Size of unpacked common modes
  std::vector<HGCROCChannelDataFrame<D> > channelData_;  ///< Array for unpacked channels
  std::vector<uint32_t> commonModeIndex_;  ///< Array for logicalMapping between unpacked channels to first common mode
  std::vector<uint16_t> commonModeData_;   ///< Array for unpacked common modes
  std::vector<uint32_t> badECOND_;         ///< Array of indices of bad ECON-Ds
};

#endif
