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

#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDPacketInfoHost.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalConfiguration.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "EventFilter/HGCalRawToDigi/interface/UnpackerTools.h"

#include <cstdint>
#include <functional>
#include <vector>

class HGCalUnpacker {
public:
  HGCalUnpacker() = default;

  // TODO @hqucms
  // define what is needed as `config`
  // HGCalUnpacker(HGCalUnpackerConfig config);

  uint8_t parseFEDData(unsigned fedId,
                       const FEDRawData& fed_data,
                       const HGCalMappingModuleIndexer& moduleIndexer,
                       const HGCalConfiguration& config,
                       hgcaldigi::HGCalDigiHost& digis,
                       hgcaldigi::HGCalECONDPacketInfoHost& econdPacketInfo,
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
  constexpr static uint16_t flags_[16] = {hgcal::DIGI_FLAG::ZS_ToA,
                                          hgcal::DIGI_FLAG::ZS_ToA_ADCm1,
                                          hgcal::DIGI_FLAG::ZS_ToA,
                                          hgcal::DIGI_FLAG::ZS_ADCm1,
                                          hgcal::DIGI_FLAG::FULL_READOUT,
                                          hgcal::DIGI_FLAG::FULL_READOUT,
                                          hgcal::DIGI_FLAG::FULL_READOUT,
                                          hgcal::DIGI_FLAG::FULL_READOUT,
                                          hgcal::DIGI_FLAG::Invalid,
                                          hgcal::DIGI_FLAG::Invalid,
                                          hgcal::DIGI_FLAG::Invalid,
                                          hgcal::DIGI_FLAG::Invalid,
                                          hgcal::DIGI_FLAG::FULL_READOUT,
                                          hgcal::DIGI_FLAG::FULL_READOUT,
                                          hgcal::DIGI_FLAG::FULL_READOUT,
                                          hgcal::DIGI_FLAG::FULL_READOUT};
  constexpr static uint32_t erxBodyBits_[16] = {24, 16, 24, 24, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32};
};

#endif
