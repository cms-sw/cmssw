#ifndef DataFormats_GEMDigi_GEMAMCStatus_h
#define DataFormats_GEMDigi_GEMAMCStatus_h
#include "GEMAMC13.h"
#include "GEMAMC.h"
#include <bitset>
#include <ostream>

class GEMAMCStatus {
public:
  union Errors {
    uint16_t ecodes;
    struct {
      uint16_t badEC : 1;  // event counter
      uint16_t badBC : 1;  // bunch crossing
      uint16_t badOC : 1;  // orbit number
      uint16_t badRunType : 1;
      uint16_t badCRC : 1;
      uint16_t MMCMlocked : 1;
      uint16_t DAQclocklocked : 1;
      uint16_t DAQnotReday : 1;
      uint16_t BC0locked : 1;
      uint16_t badFEDId : 1;
      uint16_t L1AFull : 1;
    };
  };
  union Warnings {
    uint8_t wcodes;
    struct {
      uint8_t InValidOH : 1;
      uint8_t backPressure : 1;
      uint8_t L1ANearFull : 1;
    };
  };

  GEMAMCStatus() {}
  GEMAMCStatus(const GEMAMC13* amc13, const GEMAMC& amc) {
    amcNum_ = amc.amcNum();
    Errors error{0};
    error.badEC = (amc13->lv1Id() != amc.lv1Id());
    // Last BC in AMC13 is different to TCDS, AMC, and VFAT
    error.badBC = !((amc13->bunchCrossing() == amc.bunchCrossing()) ||
                    (amc13->bunchCrossing() == 0 && amc.bunchCrossing() == GEMAMC13::lastBC));
    error.badRunType = amc.runType() != 0x1;
    // Last OC in AMC13 is different to TCDS, AMC, and VFAT
    if (amc.formatVer() == 0)
      error.badOC =
          !((uint16_t(amc13->orbitNumber()) == amc.orbitNumber()) ||
            (amc13->bunchCrossing() == 0 && uint16_t(amc.orbitNumber() + 1) == uint16_t(amc13->orbitNumber())));
    else
      error.badOC = !((amc13->orbitNumber() == (amc.orbitNumber() + 1)) ||
                      (amc13->bunchCrossing() == 0 && amc13->orbitNumber() == (amc.orbitNumber() + 2)));
    error.MMCMlocked = !amc.mmcmLocked();
    error.DAQclocklocked = !amc.daqClockLocked();
    error.DAQnotReday = !amc.daqReady();
    error.BC0locked = !amc.bc0locked();
    error.badFEDId = (amc13->sourceId() != amc.softSrcId() and amc.formatVer() != 0);
    error.L1AFull = (amc.l1aF() and amc.formatVer() != 0);
    errors_ = error.ecodes;

    Warnings warn{0};
    warn.backPressure = amc.backPressure();
    warn.L1ANearFull = (amc.l1aNF() and amc.formatVer() != 0);
    warnings_ = warn.wcodes;
  }

  void inValidOH() {
    Warnings warn{warnings_};
    warn.InValidOH = 1;
    warnings_ = warn.wcodes;
  }

  uint8_t amcNumber() const { return amcNum_; };
  bool isBad() const { return errors_ != 0; }
  uint16_t errors() const { return errors_; }
  uint8_t warnings() const { return warnings_; }

private:
  uint8_t amcNum_;
  uint16_t errors_;
  uint8_t warnings_;
};

inline std::ostream& operator<<(std::ostream& out, const GEMAMCStatus& status) {
  out << "GEMAMCStatus errors " << std::bitset<16>(status.errors()) << " warnings "
      << std::bitset<8>(status.warnings());
  return out;
}

#endif
