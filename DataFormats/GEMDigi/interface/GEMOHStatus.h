#ifndef DataFormats_GEMDigi_GEMOHStatus_h
#define DataFormats_GEMDigi_GEMOHStatus_h
#include "GEBdata.h"

// GEM OptoHyrid status
namespace gem {

    class GEMOHStatus {
    public:

    union Errors {
    uint16_t codes;
    struct {
      uint16_t InValidOptoHybrid : 1; // input link not found
      uint16_t EvtF : 1;              // Event FIFO full
      uint16_t InF : 1;               // Input FIFO full
      uint16_t L1aF : 1;              // L1A FIFO full
      uint16_t EvtSzOFW : 1;          // Event size overflow
      uint16_t Inv : 1;               // Invalid event
      uint16_t OOScAvV : 1;           // Out of Sync (EC mismatch) AMC vs VFAT
      uint16_t OOScVvV : 1;           // Out of Sync (EC mismatch) VFAT vs VFAT
      uint16_t BxmAvV : 1;            // BX mismatch AMC vs VFAT
      uint16_t BxmVvV : 1;            // 1st bit BX mismatch VFAT vs VFAT
      uint16_t InUfw : 1;             // Input FIFO underflow
      uint16_t badVFatCount : 1;
    };
    };
    union Warnings {
    uint8_t wcodes;
    struct {
      uint8_t EvtNF : 1;             // Event FIFO near full
      uint8_t InNF : 1;              // Input FIFO near full
      uint8_t L1aNF : 1;             // L1A FIFO near full
      uint8_t EvtSzW : 1;            // Event size warning
    };
    };

    GEMOHStatus(const GEBdata& oh, bool InValidOH) {
      Errors error{0};
      error.InValidOptoHybrid = InValidOH;
      error.EvtF = oh.evtF();
      error.InF = oh.inF();
      error.L1aF = oh.l1aF();
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
      warn.EvtNF = oh.evtNF();
      warn.InNF = oh.inNF();
      warn.L1aNF = oh.l1aNF();
      warn.EvtSzW = oh.evtSzW();
      warnings_ = warn.wcodes;
    }

    bool isGood() { return errors_ == 0;}
    bool isBad() { return errors_ != 0;}
    uint16_t errors() { return errors_; }
    uint8_t warnings() { return warnings_; }

    private:

    uint16_t errors_;
    uint8_t warnings_;

/*
        // check if Chamber exists.
        if (!gemROMap->isValidChamber(geb_ec)) {
          unknownChamber = true;
          LogDebug("GEMRawToDigiModule") << "InValid: amcNum " << int(amcNum) << " gebId " << int(gebId);
          continue;
        }
*/

    };
}
#endif
