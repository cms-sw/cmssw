#include "DataFormats/HcalDetId/interface/HcalDcsDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>

HcalDcsDetId::HcalDcsDetId() : HcalOtherDetId() {}

HcalDcsDetId::HcalDcsDetId(uint32_t rawid) : HcalOtherDetId(rawid) {}

HcalDcsDetId::HcalDcsDetId(DetId const& id) : HcalOtherDetId(id) {
  if ((subdet() != HcalDcsBarrel) || (subdet() != HcalDcsEndcap) || (subdet() != HcalDcsOuter) ||
      (subdet() != HcalDcsForward)) {
    throw cms::Exception("Invalid DetId") << "Cannot intialize HcalDcsDetId from " << std::hex << id_ << std::dec;
  }
}

HcalDcsDetId::HcalDcsDetId(
    HcalOtherSubdetector subd, int side_or_ring, unsigned int slc, DcsType ty, unsigned int subchan)
    : HcalOtherDetId(subd) {
  id_ |= ((side_or_ring > 0) ? ((1 << kSideOffset) | (side_or_ring << kRingOffset)) : ((-side_or_ring) << kRingOffset));
  id_ |= (slc & 0x1F) << kSliceOffset;
  id_ |= (ty & 0xF) << kTypeOffset;
  id_ |= (subchan & 0xF) << kSubChannelOffset;
}

HcalDcsDetId::DcsType HcalDcsDetId::DcsTypeFromString(const std::string& str) {
  int ty(HV);
  do {
    if (typeString(HcalDcsDetId::DcsType(ty)) == str)
      return HcalDcsDetId::DcsType(ty);
  } while (++ty != DCS_MAX);
  return DCSUNKNOWN;
}

std::string HcalDcsDetId::typeString(DcsType typ) {
  switch (typ) {
    case HV:
      return "HV";
    case BV:
      return "BV";
    case CATH:
      return "CATH";
    case DYN7:
      return "DYN7";
    case DYN8:
      return "DYN8";
    case RM_TEMP:
      return "RM_TEMP";
    case CCM_TEMP:
      return "CCM_TEMP";
    case CALIB_TEMP:
      return "CALIB_TEMP";
    case LVTTM_TEMP:
      return "LVTTM_TEMP";
    case TEMP:
      return "TEMP";
    case QPLL_LOCK:
      return "QPLL_LOCK";
    case STATUS:
      return "STATUS";
    default:
      return "DCSUNKNOWN";
  }
  return "Invalid";
}

std::ostream& operator<<(std::ostream& s, const HcalDcsDetId& id) {
  switch (id.subdet()) {
    case (HcalDcsBarrel):
      return s << "(HB" << id.zside() << ' ' << id.slice() << ' ' << id.typeString(id.type()) << id.subchannel() << ')';
    case (HcalDcsEndcap):
      return s << "(HE" << id.zside() << ' ' << id.slice() << ' ' << id.typeString(id.type()) << id.subchannel() << ')';
    case (HcalDcsOuter):
      return s << "(HO" << id.ring() << " " << id.slice() << ' ' << id.typeString(id.type()) << id.subchannel() << ')';
    case (HcalDcsForward):
      return s << "(HF" << id.zside() << ' ' << ((id.type() <= HcalDcsDetId::DYN8) ? "Q" : "") << id.slice() << ' '
               << id.typeString(id.type()) << id.subchannel() << ')';
    default:
      return s << id.rawId();
  }
}
