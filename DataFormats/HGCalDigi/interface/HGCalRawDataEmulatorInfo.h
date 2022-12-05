#ifndef DataFormats_HGCalDigi_HGCalEmulatorInfo_h
#define DataFormats_HGCalDigi_HGCalEmulatorInfo_h

#include <bitset>
#include <vector>

/**
   @class HGCalECONDEmulatorInfo
   @short This class defines a short summary of the truth information when an ECON-D data frame is generated
   It can be used to check that the unpacking outputs match the main fields.
   For the moment it stores information on the error bits and channel status
 */
class HGCalECONDEmulatorInfo {
public:
  HGCalECONDEmulatorInfo() = default;
  explicit HGCalECONDEmulatorInfo(
      bool obit, bool bbit, bool ebit, bool tbit, bool hbit, bool sbit, std::vector<uint64_t> enabled_channels = {});

  void clear();

  void addChannelsEnable(uint64_t);
  std::vector<bool> channelsEnabled(size_t) const;

  enum HGCROCEventRecoStatus { PerfectReco = 0, GoodReco = 1, FailedReco = 2, AmbiguousReco = 3 };
  HGCROCEventRecoStatus eventRecoStatus() const;

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

/**
   @class HGCalSlinkEmulatorInfo
   @short For the moment this class collects a vector of ECON-D emulator truth information
*/
class HGCalSlinkEmulatorInfo {
public:
  HGCalSlinkEmulatorInfo() = default;

  void clear();

  void addECONDEmulatedInfo(const HGCalECONDEmulatorInfo&);

private:
  std::vector<HGCalECONDEmulatorInfo> econd_info_;
};

#endif
