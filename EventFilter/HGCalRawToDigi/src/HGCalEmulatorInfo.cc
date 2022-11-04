#include "EventFilter/HGCalRawToDigi/interface/HGCalEmulatorInfo.h"

HGCalECONDEmulatorInfo::HGCalECONDEmulatorInfo(
    bool obit, bool bbit, bool ebit, bool tbit, bool hbit, bool sbit, std::vector<uint64_t> enabled_channels) {
  header_bits_[StatusBits::O] = obit;
  header_bits_[StatusBits::B] = bbit;
  header_bits_[StatusBits::E] = ebit;
  header_bits_[StatusBits::T] = tbit;
  header_bits_[StatusBits::H] = hbit;
  header_bits_[StatusBits::S] = sbit;
  for (const auto& ch_en : enabled_channels)
    pois_.emplace_back(ch_en);
}

void HGCalECONDEmulatorInfo::clear() {
  header_bits_.reset();
  pois_.clear();
}

void HGCalECONDEmulatorInfo::addChannelsEnable(uint64_t poi) { pois_.emplace_back(poi); }

std::vector<bool> HGCalECONDEmulatorInfo::channelsEnabled(size_t ch_id) const {
  std::vector<bool> ch_en;
  for (const auto& poi : pois_)
    ch_en.emplace_back(poi.test(ch_id));
  return ch_en;
}

HGCalECONDEmulatorInfo::HGCROCEventRecoStatus HGCalECONDEmulatorInfo::eventRecoStatus() const {
  return static_cast<HGCROCEventRecoStatus>(bitH() << 1 | bitT());
}

void HGCalSlinkEmulatorInfo::clear() { econd_info_.clear(); }

void HGCalSlinkEmulatorInfo::addECONDEmulatedInfo(const HGCalECONDEmulatorInfo& econd_info) {
  econd_info_.emplace_back(econd_info);
}
