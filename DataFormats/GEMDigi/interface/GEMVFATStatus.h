#ifndef DataFormats_GEMDigi_GEMVFATStatus_h
#define DataFormats_GEMDigi_GEMVFATStatus_h
#include "GEBdata.h"

namespace gem {

    class GEMVFATStatus {
    public:

    union Errors {
    uint8_t codes;
    struct {
      uint8_t InValidPosition : 1; 
      uint8_t vc : 1; // VFAT CRC error
      uint8_t InValidHeader : 1;
      uint8_t EC : 1; // does not match AMC EC
      uint8_t BC : 1; // does not match AMC BC
    };
    };
    union Warnings {
    uint8_t wcodes;
    struct {
      uint8_t basicOFW : 1;             // Basic overflow warning
      uint8_t zeroSupOFW : 1;              // Zero-sup overflow warning
    };
    };

    GEMVFATStatus(const AMCdata& amc, const VFATdata& vfat, bool inValidPosition) {
      Errors error{0};
      error.InValidPosition = inValidPosition;
      error.vc = vfat.vc();
      error.EC = vfat.ec() != amc.lv1Id();
      error.BC = vfat.bc() != amc.bunchCrossing();
        
      Warnings warn{0};
      if (vfat.header() == 0x1E)
        warn.basicOFW = 0;
      else if (vfat.header() == 0x5E)
        warn.basicOFW = 1;
      else if (vfat.header() == 0x1A)
        warn.zeroSupOFW = 0;
      else if (vfat.header() == 0x56)
        warn.zeroSupOFW = 1;
      else 
        error.InValidHeader = 1;
        
      errors_ = error.codes;
      warnings_ = warn.wcodes;

    }

    bool isGood() { return errors_ == 0;}
    bool isBad() { return errors_ != 0;}
    uint8_t errors() { return errors_; }
    uint8_t warnings() { return warnings_; }

    private:

    uint16_t errors_;
    uint8_t warnings_;


    };
}
#endif
