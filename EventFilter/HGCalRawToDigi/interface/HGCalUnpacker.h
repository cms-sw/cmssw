/****************************************************************************
 *
 * Unpacker to decode HGCal raw data from SLinks
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
#include "DataFormats/HGCalDigi/interface/HGCalFlaggedECONDInfo.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"

#include <cstdint>
#include <functional>
#include <vector>

class HGCalUnpacker {
public:
  HGCalUnpacker() {}

  // TODO @hqucms
  // define what is needed as `config`
  // HGCalUnpacker(HGCalUnpackerConfig config);

  void parseFEDData(unsigned fedId,
                    const FEDRawData& fed_data,
                    const HGCalMappingModuleIndexer& moduleIndexer,
                    hgcaldigi::HGCalDigiHost& digis,
                    std::vector<HGCalFlaggedECONDInfo>& errors,
                    bool headerOnlyMode = false);

private:
  constexpr static uint8_t tctp_[16] = {
      0b00, 0b00, 0b01, 0b00, 0b00, 0b00, 0b00, 0b00, 0b10, 0b10, 0b10, 0b10, 0b11, 0b11, 0b11, 0b11};

  constexpr static uint32_t adcm1Mask_[16] = {0b1111111111,
                                              0b0000000000,
                                              0b1111111111,
                                              0b0000000000,
                                              0b1111111111,
                                              0b1111111111,
                                              0b1111111111,
                                              0b1111111111,
                                              0b1111111111,
                                              0b1111111111,
                                              0b1111111111,
                                              0b1111111111,
                                              0b1111111111,
                                              0b1111111111,
                                              0b1111111111,
                                              0b1111111111};
  constexpr static uint32_t adcm1Shift_[16] = {
      18,
      28,
      18,
      0,
      20,
      20,
      20,
      20,
      20,
      20,
      20,
      20,
      20,
      20,
      20,
      20,
  };

  constexpr static uint32_t adcShift_[16] = {8, 18, 8, 18, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0};
  constexpr static uint32_t adcMask_[16] = {0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b0000000000,
                                            0b0000000000,
                                            0b0000000000,
                                            0b0000000000};

  constexpr static uint32_t totShift_[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10};
  constexpr static uint32_t totMask_[16] = {0b0000000000,
                                            0b0000000000,
                                            0b0000000000,
                                            0b0000000000,
                                            0b0000000000,
                                            0b0000000000,
                                            0b0000000000,
                                            0b0000000000,
                                            0b0000000000,
                                            0b0000000000,
                                            0b0000000000,
                                            0b0000000000,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111};

  constexpr static uint32_t toaShift_[16] = {0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  constexpr static uint32_t toaMask_[16] = {0b0000000000,
                                            0b0000000000,
                                            0b0000000000,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111,
                                            0b1111111111};

  constexpr static uint32_t erxBodyBits_[16] = {24, 16, 24, 24, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32};

  // HGCalUnpackerConfig config_;
  // size_t channelDataSize_{0};                                             ///< Size of unpacked channels
  // size_t commonModeDataSize_{0};                                          ///< Size of unpacked common modes
  // std::vector<HGCROCChannelDataFrame<HGCalElectronicsId> > channelData_;  ///< Array for unpacked channels
  // std::vector<uint16_t> commonModeSum_;
  // std::vector<HGCROCChannelDataFrame<HGCalElectronicsId> > commonModeData_;  ///< Array for unpacked common modes
  // HGCalFlaggedECONDInfoCollection flaggedECOND_;                             ///< Array with flagged ECON-D information
};

#endif
