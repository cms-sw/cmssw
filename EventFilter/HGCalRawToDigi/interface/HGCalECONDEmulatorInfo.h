#ifndef EventFilter_HGCalRawToDigi_HGCalECONDEmulatorInfo_h
#define EventFilter_HGCalRawToDigi_HGCalECONDEmulatorInfo_h

#include <bitset>
#include <vector>

class HGCalECONDEmulatorInfo {
public:
  HGCalECONDEmulatorInfo() = default;
  explicit HGCalECONDEmulatorInfo(
      bool obit, bool bbit, bool ebit, bool tbit, bool hbit, bool sbit, std::vector<uint64_t> enabled_channels = {}) {
    header_bits_[StatusBits::O] = obit;
    header_bits_[StatusBits::B] = bbit;
    header_bits_[StatusBits::E] = ebit;
    header_bits_[StatusBits::T] = tbit;
    header_bits_[StatusBits::H] = hbit;
    header_bits_[StatusBits::S] = sbit;
    for (const auto& ch_en : enabled_channels)
      pois_.emplace_back(ch_en);
  }

  void addChannelsEnable(uint64_t poi) { pois_.emplace_back(poi); }
  std::vector<bool> channelsEnabled(size_t ch_id) const {
    std::vector<bool> ch_en;
    for (const auto& poi : pois_)
      ch_en.emplace_back(poi.test(ch_id));
    return ch_en;
  }

  enum HGCROCEventRecoStatus { PerfectReco = 0, GoodReco = 1, FailedReco = 2, AmbiguousReco = 3 };
  HGCROCEventRecoStatus eventRecoStatus() const { return static_cast<HGCROCEventRecoStatus>(bitH() << 1 | bitT()); }

  bool bitO() const { return header_bits_.test(StatusBits::O); }
  bool bitB() const { return header_bits_.test(StatusBits::B); }
  bool bitE() const { return header_bits_.test(StatusBits::E); }
  bool bitT() const { return header_bits_.test(StatusBits::T); }
  bool bitH() const { return header_bits_.test(StatusBits::H); }
  bool bitS() const { return header_bits_.test(StatusBits::S); }

private:
  enum StatusBits { O = 0, B, E, T, H, S };
  std::bitset<6> header_bits_;
  std::vector<std::bitset<37> > pois_;
};

#endif
