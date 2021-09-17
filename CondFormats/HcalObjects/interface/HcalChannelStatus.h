#ifndef HcalChannelStatus_h
#define HcalChannelStatus_h

/* 
\class HcalChannelStatus
\author Radek Ofierzynski
contains one channel status and corresponding DetId
*/

#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

class HcalChannelStatus {
public:
  // contains the defined bits for easy access, see https://twiki.cern.ch/twiki/bin/view/CMS/HcalDataValidationWorkflow
  enum StatusBit {
    HcalCellOff = 0,   // 1=Hcal cell is off
    HcalCellMask = 1,  // 1=Hcal cell is masked/to be masked at RecHit Level
    // Quality Bits
    HcalCellDead = 5,                         // 1=Hcal cell is dead (from DQM algo)
    HcalCellHot = 6,                          // 1=Hcal cell is hot (from DQM algo)
    HcalCellStabErr = 7,                      // 1=Hcal cell has stability error
    HcalCellTimErr = 8,                       // 1=Hcal cell has timing error
    HcalCellExcludeFromHBHENoiseSummary = 9,  //1=cell is completely excluded from all HBHENoiseSummary computations
    HcalCellExcludeFromHBHENoiseSummaryR45 =
        10,  //1=cell's rechit is excluded when calculating the TS4TS5 ratio only in the HBHE Noise Summary
    HcalBadLaserSignal = 11,  //1 = channel does not receive good laser calibration signal
    // Trigger Bits
    HcalCellTrigMask = 15,  // 1=cell is masked from the Trigger
    // CaloTower Bits
    HcalCellCaloTowerMask = 18,  // 1=cell is always excluded from the CaloTower, regardless of other bit settings.
    HcalCellCaloTowerProb = 19   // 1=cell is counted as problematic within the tower.
  };

  HcalChannelStatus() : mId(0), mStatus(0) {}
  HcalChannelStatus(unsigned long fid, uint32_t status) : mId(fid), mStatus(status) {}

  //  void setDetId(unsigned long fid) {mId = fid;}
  void setValue(uint32_t value) { mStatus = value; }

  // for the following, one can use unsigned int values or HcalChannelStatus::StatusBit values
  //   e.g. 5 or HcalChannelStatus::HcalCellDead
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
