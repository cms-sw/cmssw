/****************************************************************************
 *
 * This is a part of HGCAL offline software.
 * Authors:
 *   Laurent Forthomme, CERN
 *   Pedro Silva, CERN
 *
 ****************************************************************************/

#include "DataFormats/HGCalDigi/interface/HGCalRawDataEmulatorInfo.h"

//-----------------------------------------------
// ECON-D emulator info
//-----------------------------------------------

HGCalECONDEmulatorInfo::HGCalECONDEmulatorInfo(bool obit,
                                               bool bbit,
                                               bool ebit,
                                               bool tbit,
                                               bool hbit,
                                               bool sbit,
                                               const std::vector<std::vector<bool> >& enabled_channels) {
  header_bits_[StatusBits::O] = obit;
  header_bits_[StatusBits::B] = bbit;
  header_bits_[StatusBits::E] = ebit;
  header_bits_[StatusBits::T] = tbit;
  header_bits_[StatusBits::H] = hbit;
  header_bits_[StatusBits::S] = sbit;
  for (const auto& ch_en : enabled_channels)
    erx_pois_.emplace_back(ch_en);
}

void HGCalECONDEmulatorInfo::clear() {
  header_bits_.reset();
  erx_pois_.clear();
}

void HGCalECONDEmulatorInfo::addERxChannelsEnable(const std::vector<bool>& erx_channels_poi) {
  erx_pois_.emplace_back(erx_channels_poi);
}

std::vector<bool> HGCalECONDEmulatorInfo::channelsEnabled(size_t ch_id) const {
  std::vector<bool> ch_en;
  for (const auto& erx_channels_poi : erx_pois_)
    ch_en.emplace_back(erx_channels_poi.at(ch_id));
  return ch_en;
}

HGCalECONDEmulatorInfo::HGCROCEventRecoStatus HGCalECONDEmulatorInfo::eventRecoStatus() const {
  return static_cast<HGCROCEventRecoStatus>(bitH() << 1 | bitT());
}

//-----------------------------------------------
// Capture block emulator info
//-----------------------------------------------

void HGCalCaptureBlockEmulatorInfo::addECONDEmulatedInfo(unsigned int econd_id,
                                                         const HGCalECONDEmulatorInfo& econd_info) {
  econd_info_[econd_id] = econd_info;
}

//-----------------------------------------------
// S-link emulator info
//-----------------------------------------------

void HGCalSlinkEmulatorInfo::addCaptureBlockEmulatedInfo(unsigned int cb_id,
                                                         const HGCalCaptureBlockEmulatorInfo& cb_info) {
  cb_info_[cb_id] = cb_info;
}

HGCalCaptureBlockEmulatorInfo& HGCalSlinkEmulatorInfo::captureBlockEmulatedInfo(unsigned int cb_id) {
  return cb_info_[cb_id];
}
