#ifndef DataFormats_GEMDigi_GEMOptoHybrid_h
#define DataFormats_GEMDigi_GEMOptoHybrid_h
#include "GEMVFAT.h"
#include <vector>

class GEMOptoHybrid {
public:
  union GEBchamberHeader {
    uint64_t word;
    // v301 dataformat
    struct {
      uint64_t : 10;            // unused
      uint64_t BxmVvV : 1;      // 1st bit BX mismatch VFAT vs VFAT
      uint64_t BxmAvV : 1;      // BX mismatch AMC vs VFAT
      uint64_t OOScVvV : 1;     // Out of Sync (EC mismatch) VFAT vs VFAT
      uint64_t OOScAvV : 1;     // Out of Sync (EC mismatch) AMC vs VFAT
      uint64_t Inv : 1;         // Invalid event
      uint64_t EvtSzW : 1;      // Event size warning
      uint64_t L1aNF : 1;       // L1A FIFO near full
      uint64_t InNF : 1;        // Input FIFO near full
      uint64_t EvtNF : 1;       // Event FIFO near full
      uint64_t EvtSzOFW : 1;    // Event size overflow
      uint64_t L1aF : 1;        // L1A FIFO full
      uint64_t InF : 1;         // Input FIFO full
      uint64_t EvtF : 1;        // Event FIFO full
      uint64_t VfWdCnt : 12;    // VFAT word count (in number of 64-bit words)
      uint64_t InputID : 5;     // Input link ID
      uint64_t CALIB_CHAN : 7;  // Calibration channel number
      uint64_t : 17;            // unused
    };
    // v302 dataformat
    struct {
      uint64_t : 10;                // unused
      uint64_t BxmVvVv302 : 1;      // 1st bit BX mismatch VFAT vs VFAT
      uint64_t BxmAvVv302 : 1;      // BX mismatch AMC vs VFAT
      uint64_t OOScVvVv302 : 1;     // Out of Sync (EC mismatch) VFAT vs VFAT
      uint64_t OOScAvVv302 : 1;     // Out of Sync (EC mismatch) AMC vs VFAT
      uint64_t InvV302 : 1;         // Invalid event
      uint64_t EvtSzWv302 : 1;      // Event size warning
      uint64_t : 1;                 // unused
      uint64_t InNFv302 : 1;        // Input FIFO near full
      uint64_t EvtNFv302 : 1;       // Event FIFO near full
      uint64_t EvtSzOFWv302 : 1;    // Event size overflow
      uint64_t : 1;                 // unused
      uint64_t InFv302 : 1;         // Input FIFO full
      uint64_t EvtFv302 : 1;        // Event FIFO full
      uint64_t VfWdCntV302 : 12;    // VFAT word count (in number of 64-bit words)
      uint64_t InputIDv302 : 5;     // Input link ID
      uint64_t CALIB_CHANv302 : 7;  // Calibration channel number
      uint64_t : 17;                // unused
    };
  };

  union GEBchamberTrailer {
    uint64_t word;
    // v301 dataformat
    struct {
      uint64_t ecOH : 20;      // NOT USED - OptoHybrid event counter
      uint64_t bcOH : 13;      // NOT USED - OptoHybrid bunch crossing
      uint64_t InUfw : 1;      // Input FIFO underflow
      uint64_t SkD : 1;        // NOT USED - Stuck data
      uint64_t EvUfw : 1;      // NOT USED - Event FIFO underflow
      uint64_t VfWdCntT : 12;  // VFAT word count (in number of 64-bit words)
      uint64_t crc16 : 16;     // CRC of OptoHybrid data (currently not available – filled with 0)
    };
    // v302 dataformat
    struct {
      uint64_t VFATMask : 24;      // Enabled VFAT, Set 1 if the VFAT is enabled
      uint64_t ZSMask : 24;        // Zero-suppressed VFAT, Set 1 if the VFAT payload has been zero-suppressed
      uint64_t : 3;                // unused
      uint64_t InUfwV302 : 1;      // Input FIFO underflow
      uint64_t VfWdCntTv302 : 12;  // VFAT word count (in number of 64-bit words)
    };
  };

  GEMOptoHybrid() : ch_(0), ct_(0){};
  ~GEMOptoHybrid() { vfatd_.clear(); }

  void setVersion(uint8_t i) { ver_ = i; }
  uint8_t version() const { return ver_; }

  //!Read chamberHeader from the block.
  void setChamberHeader(uint64_t word) { ch_ = word; }
  void setChamberHeader(uint16_t vfatWordCnt, uint8_t inputID) {
    GEBchamberHeader u{0};
    u.VfWdCnt = vfatWordCnt;
    u.InputID = inputID;
    ch_ = u.word;
  }
  uint64_t getChamberHeader() const { return ch_; }

  //!Read chamberTrailer from the block.
  void setChamberTrailer(uint64_t word) { ct_ = word; }
  void setChamberTrailer(uint32_t ecOH, uint16_t bcOH, uint16_t vfatWordCntT) {
    GEBchamberTrailer u{0};
    u.ecOH = ecOH;
    u.bcOH = bcOH;
    u.VfWdCntT = vfatWordCntT;
    ct_ = u.word;
  }
  uint64_t getChamberTrailer() const { return ct_; }

  // v301
  uint16_t vfatWordCnt() const {
    if (ver_ == 0)
      return GEBchamberHeader{ch_}.VfWdCnt;
    return GEBchamberHeader{ch_}.VfWdCntV302;
  }
  uint8_t inputID() const {
    if (ver_ == 0)
      return GEBchamberHeader{ch_}.InputID;
    return GEBchamberHeader{ch_}.InputIDv302;
  }
  uint16_t vfatWordCntT() const {
    if (ver_ == 0)
      return GEBchamberTrailer{ct_}.VfWdCntT;
    return GEBchamberTrailer{ch_}.VfWdCntTv302;
  }

  bool bxmVvV() const { return GEBchamberHeader{ch_}.BxmVvV; }
  bool bxmAvV() const { return GEBchamberHeader{ch_}.BxmAvV; }
  bool oOScVvV() const { return GEBchamberHeader{ch_}.OOScVvV; }
  bool oOScAvV() const { return GEBchamberHeader{ch_}.OOScAvV; }
  bool inv() const { return GEBchamberHeader{ch_}.Inv; }
  bool evtSzW() const { return GEBchamberHeader{ch_}.EvtSzW; }
  bool inNF() const { return GEBchamberHeader{ch_}.InNF; }
  bool evtNF() const { return GEBchamberHeader{ch_}.EvtNF; }
  bool evtSzOFW() const { return GEBchamberHeader{ch_}.EvtSzOFW; }
  bool inF() const { return GEBchamberHeader{ch_}.InF; }
  bool evtF() const { return GEBchamberHeader{ch_}.EvtF; }
  bool inUfw() const {
    if (ver_ == 0)
      return GEBchamberTrailer{ct_}.InUfw;
    return GEBchamberTrailer{ct_}.InUfwV302;
  }

  bool noVFAT() const { return false; }     // to be removed
  bool stuckData() const { return false; }  // to be removed
  bool evUfw() const { return false; }      // to be removed

  // v301
  bool l1aNF() const { return GEBchamberHeader{ch_}.L1aNF; }
  bool l1aF() const { return GEBchamberHeader{ch_}.L1aF; }

  // v302
  uint32_t vfatMask() const { return GEBchamberTrailer{ct_}.VFATMask; }
  uint32_t zsMask() const { return GEBchamberTrailer{ct_}.ZSMask; }

  //!Adds VFAT data to the vector
  void addVFAT(GEMVFAT v) { vfatd_.push_back(v); }
  //!Returns the vector of VFAT data
  const std::vector<GEMVFAT>* vFATs() const { return &vfatd_; }
  //!Clear the vector rof VFAT data
  void clearVFATs() { vfatd_.clear(); }

  static const int sizeGebID = 5;

private:
  uint8_t ver_;  // Data Format version

  uint64_t ch_;  // GEBchamberHeader
  uint64_t ct_;  // GEBchamberTrailer

  std::vector<GEMVFAT> vfatd_;
};
#endif
