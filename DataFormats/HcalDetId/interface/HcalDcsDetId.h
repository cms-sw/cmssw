// -*- C++ -*-
#ifndef DataFormats_HcalDetId_HcalDcsDetId_h
#define DataFormats_HcalDetId_HcalDcsDetId_h

#include <iosfwd>
//#include <string>
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

/** \class HcalDcsDetId

for use to identify HcalDcsChannels

bit packing
[31:28] from DetId to identify detector (Hcal)
[27:25] from DetId to identify subdetector (HcalOther)
[24:20] other subdet id
[19:19] zside +/-
[18:17] HO ring (not important in other subdets)
[16:12] Slice (phi slice for HB HE, Sector for HO, Quadrant for HF)
[11:8]  Type from the DCSType list
[7:4]   sub-channel a number to identify the channel can be from 0 to 15
[3:0]   still open

 */

class HcalDcsDetId : public HcalOtherDetId {
public:
  enum DcsType {
    HV = 1,
    BV = 2,
    CATH = 3,
    DYN7 = 4,
    DYN8 = 5,
    RM_TEMP = 6,
    CCM_TEMP = 7,
    CALIB_TEMP = 8,
    LVTTM_TEMP = 9,
    TEMP = 10,
    QPLL_LOCK = 11,
    STATUS = 12,
    DCSUNKNOWN = 15,
    DCS_MAX = 16
  };

  HcalDcsDetId();
  HcalDcsDetId(uint32_t rawid);
  HcalDcsDetId(const DetId& id);
  HcalDcsDetId(HcalOtherSubdetector subd, int side_or_ring, unsigned int slc, DcsType ty, unsigned int subchan);

  static DcsType DcsTypeFromString(const std::string& str);
  static std::string typeString(DcsType typ);

  int zside() const { return (((id_ >> kSideOffset) & 0x1) ? 1 : -1); }
  int ring() const { return zside() * ((id_ >> kRingOffset) & 0x3); }
  int slice() const { return ((id_ >> kSliceOffset) & 0x1F); }
  DcsType type() const { return DcsType((id_ >> kTypeOffset) & 0xF); }
  int subchannel() const { return ((id_ >> kSubChannelOffset) & 0xF); }

  static const int maxLinearIndex = 0x16800;

protected:
  static unsigned int const kSideOffset = 19;
  static unsigned int const kRingOffset = 17;
  static unsigned int const kSliceOffset = 12;
  static unsigned int const kTypeOffset = 8;
  static unsigned int const kSubChannelOffset = 4;
};

std::ostream& operator<<(std::ostream&, const HcalDcsDetId& id);

#endif
