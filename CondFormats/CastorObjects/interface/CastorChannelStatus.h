#ifndef CastorChannelStatus_h
#define CastorChannelStatus_h

/* 
\class CastorChannelStatus
\author Radek Ofierzynski
contains one channel status and corresponding DetId
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <cstdint>

class CastorChannelStatus {
public:
  // contains the defined bits for easy access, see https://twiki.cern.ch/twiki/bin/view/CMS/CastorDataValidationWorkflow
  /* Original Hcal stuff
  enum StatusBit {       
    HcalCellOff=0,      // 1=Hcal cell is off
    HcalCellL1Mask=1,   // 1=Hcal cell is masked/to be masked by L1 trigger
    HcalCellDead=5,     // 1=Hcal cell is dead (from DQM algo)
    HcalCellHot=6,      // 1=Hcal cell is hot (from DQM algo)
    HcalCellStabErr=7,  // 1=Hcal cell has stability error
    HcalCellTimErr=8    // 1=Hcal cell has timing error
  };*/
  enum StatusBit { UNKNOWN = 0, BAD = 1, GOOD = 2, HOT = 3, DEAD = 4, END = 5 };

  CastorChannelStatus() : mId(0), mStatus(0) {}
  CastorChannelStatus(unsigned long fid, uint32_t status) : mId(fid), mStatus(status) {}
  CastorChannelStatus(unsigned long fid, std::string status) : mId(fid) {
    if (status == "BAD")
      mStatus = BAD;
    else if (status == "GOOD")
      mStatus = GOOD;
    else if (status == "HOT")
      mStatus = HOT;
    else if (status == "DEAD")
      mStatus = DEAD;
    else if (status == "END")
      mStatus = END;
    else
      mStatus = UNKNOWN;
  }

  //  void setDetId(unsigned long fid) {mId = fid;}
  void setValue(uint32_t value) { mStatus = value; }

  // for the following, one can use unsigned int values or CastorChannelStatus::StatusBit values
  //   e.g. 4 or CastorChannelStatus::DEAD
  void setBit(unsigned int bitnumber) {
    uint32_t statadd = 0x1 << (bitnumber);
    mStatus = mStatus | statadd;
  }
  void unsetBit(unsigned int bitnumber) {
    uint32_t statadd = 0x1 << (bitnumber);
    statadd = ~statadd;
    mStatus = mStatus & statadd;
  }

  bool isBitSet(unsigned int bitnumber) const {
    uint32_t statadd = 0x1 << (bitnumber);
    return (mStatus & statadd) ? (true) : (false);
  }

  uint32_t rawId() const { return mId; }

  uint32_t getValue() const { return mStatus; }

private:
  uint32_t mId;
  uint32_t mStatus;

  COND_SERIALIZABLE;
};
#endif
