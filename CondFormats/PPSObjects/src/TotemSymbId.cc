/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@cern.ch)
*
****************************************************************************/

#include "CondFormats/PPSObjects/interface/TotemSymbId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemT2DetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"

void TotemSymbID::print(std::ostream& os, std::string subSystemName) const {
  const auto ctppsDetId = CTPPSDetId(symbolicID);
  os << "DetId=" << symbolicID << " (";

  if (subSystemName == "TrackingStrip") {
    os << "subDet=" << ctppsDetId.subdetId() << " " << TotemRPDetId(symbolicID);
  } else if (subSystemName == "TimingDiamond") {
    os << "subDet=" << ctppsDetId.subdetId() << " " << CTPPSDiamondDetId(symbolicID);
  } else if (subSystemName == "TotemT2") {
    os << "subDet=" << ctppsDetId.subdetId() << " " << TotemT2DetId(symbolicID);
  } else if (subSystemName == "TotemTiming") {
    const auto timingId = TotemTimingDetId(symbolicID);
    os << "subDet=" << ctppsDetId.subdetId() << " " << timingId;
    if (timingId.channel() == TotemTimingDetId::ID_NOT_SET || timingId.plane() == 0) {
      os << ") (default plane value:" << 0 << ", detector id not set value: " << TotemTimingDetId::ID_NOT_SET;
    }
  } else {
    os << ctppsDetId;
  }

  os << ")";
}

std::ostream& operator<<(std::ostream& s, const TotemSymbID& sid) {
  s << "DetId=" << sid.symbolicID << " (" << CTPPSDetId(sid.symbolicID) << ")";

  return s;
}
