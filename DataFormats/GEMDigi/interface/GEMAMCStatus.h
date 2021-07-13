#ifndef DataFormats_GEMDigi_GEMAMCStatus_h
#define DataFormats_GEMDigi_GEMAMCStatus_h
#include "AMC13Event.h"
#include "AMCdata.h"

namespace gem {

    class GEMAMCStatus {
    public:
    union Errors {
    uint16_t ecodes;
    struct {
      uint16_t InvalidAMC : 1;
      uint16_t badEC : 1; // event counter
      uint16_t badBC : 1; // bunch crossing
      uint16_t badOC : 1; // orbit number
      uint16_t badRunType : 1;
      uint16_t badCRC : 1;
      uint16_t MMCMlocked : 1; 
      uint16_t DAQclocklocked : 1;
      uint16_t DAQnotReday : 1;
      uint16_t BC0locked : 1;
      uint16_t InvalidAMCSize : 1;
    };
    };
    union Warnings {
    uint8_t wcodes;
    struct {
      uint8_t backPressure : 1;
    };
    };

    GEMAMCStatus(const AMC13Event* amc13, const AMCdata& amc, int isValidAMC) {
        Errors error{0};
        error.InvalidAMC = !isValidAMC;
        error.badEC = (amc13->lv1Id() != amc.lv1Id());
        error.badBC = (amc13->bunchCrossing() != amc.bunchCrossing());
        error.badRunType = amc.runType() != 0x1; 
        error.badOC = (amc13->orbitNumber() != amc.orbitNumber());
        error.badCRC = (amc13->crc() != amc.crc());
        error.MMCMlocked = !amc.mmcmLocked();
        error.DAQclocklocked = !amc.daqClockLocked();
        error.DAQnotReday = !amc.daqReady();
        error.BC0locked = !amc.bc0locked();
        //error.InvalidAMCSize = amc13->getAMCsize(i) != amc.dataLength();
        errors_ = error.ecodes;

        Warnings warn{0};
        warn.backPressure = amc.backPressure();
        warnings_ = warn.wcodes;
    }

    bool isGood() { return errors_ == 0;}
    bool isBad() { return errors_ != 0;}
    uint16_t errors() { return errors_; }
    uint8_t warnings() { return warnings_; }

    private:

    uint16_t errors_;
    uint8_t warnings_;

    };
}
#endif
