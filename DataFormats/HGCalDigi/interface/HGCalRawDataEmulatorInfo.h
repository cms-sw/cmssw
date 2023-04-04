/****************************************************************************
 *
 * This is a part of HGCAL offline software.
 * Authors:
 *   Laurent Forthomme, CERN
 *   Pedro Silva, CERN
 *
 ****************************************************************************/

#ifndef DataFormats_HGCalDigi_HGCalRawDataEmulatorInfo_h
#define DataFormats_HGCalDigi_HGCalRawDataEmulatorInfo_h

#include <bitset>
#include <unordered_map>
#include <vector>

/// Short summary of the truth information when an ECON-D data frame is generated
/// \note It can be used to check that the unpacking outputs match the main fields.
///   For the moment it stores information on the error bits and channel status
class HGCalECONDEmulatorInfo {
public:
  HGCalECONDEmulatorInfo() = default;
  explicit HGCalECONDEmulatorInfo(bool, bool, bool, bool, bool, bool, const std::vector<std::vector<bool> >& = {});

  void clear();

  void addERxChannelsEnable(const std::vector<bool>&);
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
  std::vector<std::vector<bool> > erx_pois_;
};

/// Map of ECON-D emulator truth information within a capture block
class HGCalCaptureBlockEmulatorInfo {
public:
  HGCalCaptureBlockEmulatorInfo() = default;

  inline void clear() { econd_info_.clear(); }

  void addECONDEmulatedInfo(unsigned int, const HGCalECONDEmulatorInfo&);

private:
  std::unordered_map<unsigned int, HGCalECONDEmulatorInfo> econd_info_;
};

/// Map of capture block emulator truth information within a S-link payload
class HGCalSlinkEmulatorInfo {
public:
  HGCalSlinkEmulatorInfo() = default;

  inline void clear() { cb_info_.clear(); }

  void addCaptureBlockEmulatedInfo(unsigned int, const HGCalCaptureBlockEmulatorInfo&);
  HGCalCaptureBlockEmulatorInfo& captureBlockEmulatedInfo(unsigned int);

private:
  std::unordered_map<unsigned int, HGCalCaptureBlockEmulatorInfo> cb_info_;
};

#endif
