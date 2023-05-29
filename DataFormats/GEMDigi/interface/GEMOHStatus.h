#ifndef DataFormats_GEMDigi_GEMOHStatus_h
#define DataFormats_GEMDigi_GEMOHStatus_h
#include "GEMOptoHybrid.h"
#include <bitset>
#include <ostream>

// GEM OptoHybrid status
class GEMOHStatus {
public:
  union Errors {
    uint16_t codes;
    struct {
      uint16_t EvtF : 1;      // Event FIFO full
      uint16_t InF : 1;       // Input FIFO full
      uint16_t L1aF : 1;      // L1A FIFO full
      uint16_t EvtSzOFW : 1;  // Event size overflow
      uint16_t Inv : 1;       // Invalid event
      uint16_t OOScAvV : 1;   // Out of Sync (EC mismatch) AMC vs VFAT
      uint16_t OOScVvV : 1;   // Out of Sync (EC mismatch) VFAT vs VFAT
      uint16_t BxmAvV : 1;    // BX mismatch AMC vs VFAT
      uint16_t BxmVvV : 1;    // 1st bit BX mismatch VFAT vs VFAT
      uint16_t InUfw : 1;     // Input FIFO underflow
      uint16_t badVFatCount : 1;
    };
  };
  union Warnings {
    uint8_t wcodes;
    struct {
      uint8_t EvtNF : 1;   // Event FIFO near full
      uint8_t InNF : 1;    // Input FIFO near full
      uint8_t L1aNF : 1;   // L1A FIFO near full
      uint8_t EvtSzW : 1;  // Event size warning
      uint8_t InValidVFAT : 1;
      uint8_t missingVFAT : 1;
    };
  };

  GEMOHStatus() {}
  GEMOHStatus(const GEMOptoHybrid& oh, int chamberType) : chamberType_(chamberType) {
    Errors error{0};
    error.EvtF = oh.evtF();
    error.InF = oh.inF();
    error.L1aF = (oh.l1aF() and (oh.version() == 0));
    error.EvtSzOFW = oh.evtSzOFW();
    error.Inv = oh.inv();
    error.OOScAvV = oh.oOScAvV();
    error.OOScVvV = oh.oOScVvV();
    error.BxmAvV = oh.bxmAvV();
    error.BxmVvV = oh.bxmVvV();
    error.InUfw = oh.inUfw();
    error.badVFatCount = oh.vfatWordCnt() != oh.vfatWordCntT();
    errors_ = error.codes;

    Warnings warn{0};
    existVFATs_ = oh.existVFATs();
    vfatMask_ = oh.vfatMask();
    zsMask_ = oh.zsMask();
    missingVFATs_ = (existVFATs_ ^ 0xffffff) & (vfatMask_ & (zsMask_ ^ 0xffffff));
    warn.EvtNF = oh.evtNF();
    warn.InNF = oh.inNF();
    warn.L1aNF = (oh.l1aNF() and (oh.version() == 0));
    warn.EvtSzW = oh.evtSzW();
    warn.missingVFAT = (oh.version() != 0) and (missingVFATs_ != 0);
    warnings_ = warn.wcodes;
  }

  void inValidVFAT() {
    Warnings warn{warnings_};
    warn.InValidVFAT = 1;
    warnings_ = warn.wcodes;
  }

  bool isBad() const { return errors_ != 0; }
  uint16_t errors() const { return errors_; }
  uint8_t warnings() const { return warnings_; }
  uint32_t missingVFATs() const { return missingVFATs_; }
  uint32_t vfatMask() const { return vfatMask_; }
  uint32_t zsMask() const { return zsMask_; }
  uint32_t existVFATs() const { return existVFATs_; }
  int chamberType() const { return chamberType_; }

private:
  int chamberType_;
  uint16_t errors_;
  uint8_t warnings_;
  uint32_t missingVFATs_;
  uint32_t vfatMask_;
  uint32_t zsMask_;
  uint32_t existVFATs_;
};

inline std::ostream& operator<<(std::ostream& out, const GEMOHStatus& status) {
  out << "GEMOHStatus errors " << std::bitset<16>(status.errors()) << " warnings " << std::bitset<8>(status.warnings());
  return out;
}

#endif
