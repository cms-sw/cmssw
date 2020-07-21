#ifndef DataFormats_GEMDigi_GEBdata_h
#define DataFormats_GEMDigi_GEBdata_h
#include "VFATdata.h"
#include <vector>

namespace gem {
  // Input status 1 bit for each
  // BX mismatch GLIB OH / BX mismatch GLIB VFAT / OOS GLIB OH / OOS GLIB VFAT / No VFAT marker
  // Event size warn / L1AFIFO near full / InFIFO near full / EvtFIFO near full / Event size overflow
  // L1AFIFO full / InFIFO full / EvtFIFO full
  union GEBchamberHeader {
    uint64_t word;
    struct {
      uint64_t BxmVvV : 11;           // 1st bit BX mismatch VFAT vs VFAT
      uint64_t BxmAvV : 1;            // BX mismatch AMC vs VFAT
      uint64_t OOScVvV : 1;           // OOS (EC mismatch) VFAT vs VFAT
      uint64_t OOScAvV : 1;           // OOS (EC mismatch) AMC vs VFAT
      uint64_t noVFAT : 1;            // No VFAT marker
      uint64_t EvtSzW : 1;            // Event size warning
      uint64_t L1aNF : 1;             // L1A FIFO near full
      uint64_t InNF : 1;              // Input FIFO near full
      uint64_t EvtNF : 1;             // Event FIFO near full
      uint64_t EvtSzOFW : 1;          // Event size overflow
      uint64_t L1aF : 1;              // L1A FIFO full
      uint64_t InF : 1;               // Input FIFO full
      uint64_t EvtF : 1;              // Event FIFO full
      uint64_t VfWdCnt : 12;          // VFAT word count (in number of 64-bit words)
      uint64_t inputID : 5;           // Input link ID
      uint64_t zeroSupWordsCnt : 24;  // Number of zero suppressed VFAT 64bit words
    };
  };
  union GEBchamberTrailer {
    uint64_t word;
    struct {
      uint64_t ecOH : 20;      // OH event counter
      uint64_t bcOH : 13;      // OH bunch crossing
      uint64_t InUfw : 1;      // Input FIFO underflow
      uint64_t SkD : 1;        // Stuck data
      uint64_t EvUfw : 1;      // Event FIFO underflow
      uint64_t VfWdCntT : 12;  // VFAT word count (in number of 64-bit words)
      uint64_t crc16 : 16;     // CRC of OH data (currently not available – filled with 0)
    };
  };

  class GEBdata {
  public:
    GEBdata() : ch_(0), ct_(0){};
    ~GEBdata() { vfatd_.clear(); }

    //!Read chamberHeader from the block.
    void setChamberHeader(uint64_t word) { ch_ = word; }
    void setChamberHeader(uint16_t vfatWordCnt, uint8_t inputID) {
      GEBchamberHeader u{0};
      u.VfWdCnt = vfatWordCnt;
      u.inputID = inputID;
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

    uint16_t bxmVvV() const { return GEBchamberHeader{ch_}.BxmVvV; }
    uint8_t bxmAvV() const { return GEBchamberHeader{ch_}.BxmAvV; }
    uint8_t oOScVvV() const { return GEBchamberHeader{ch_}.OOScVvV; }
    uint8_t oOScAvV() const { return GEBchamberHeader{ch_}.OOScAvV; }
    uint8_t noVFAT() const { return GEBchamberHeader{ch_}.noVFAT; }
    uint8_t evtSzW() const { return GEBchamberHeader{ch_}.EvtSzW; }
    uint8_t l1aNF() const { return GEBchamberHeader{ch_}.L1aNF; }
    uint8_t inNF() const { return GEBchamberHeader{ch_}.InNF; }
    uint8_t evtNF() const { return GEBchamberHeader{ch_}.EvtNF; }
    uint8_t evtSzOFW() const { return GEBchamberHeader{ch_}.EvtSzOFW; }
    uint8_t l1aF() const { return GEBchamberHeader{ch_}.L1aF; }
    uint8_t inF() const { return GEBchamberHeader{ch_}.InF; }
    uint8_t evtF() const { return GEBchamberHeader{ch_}.EvtF; }
    uint16_t vfatWordCnt() const { return GEBchamberHeader{ch_}.VfWdCnt; }
    uint8_t inputID() const { return GEBchamberHeader{ch_}.inputID; }
    uint32_t zeroSupWordsCnt() const { return GEBchamberHeader{ch_}.zeroSupWordsCnt; }

    uint32_t ecOH() const { return GEBchamberTrailer{ct_}.ecOH; }
    uint16_t bcOH() const { return GEBchamberTrailer{ct_}.bcOH; }
    uint8_t inUfw() const { return GEBchamberTrailer{ct_}.InUfw; }
    uint8_t stuckData() const { return GEBchamberTrailer{ct_}.SkD; }
    uint8_t evUfw() const { return GEBchamberTrailer{ct_}.EvUfw; }
    uint16_t vfatWordCntT() const { return GEBchamberTrailer{ct_}.VfWdCntT; }
    uint16_t crc() const { return GEBchamberTrailer{ct_}.crc16; }

    //!Adds VFAT data to the vector
    void addVFAT(VFATdata v) { vfatd_.push_back(v); }
    //!Returns the vector of VFAT data
    const std::vector<VFATdata>* vFATs() const { return &vfatd_; }
    //!Clear the vector rof VFAT data
    void clearVFATs() { vfatd_.clear(); }

    static const int sizeGebID = 5;

  private:
    uint64_t ch_;  // GEBchamberHeader
    uint64_t ct_;  // GEBchamberTrailer

    std::vector<VFATdata> vfatd_;
  };
}  // namespace gem
#endif
