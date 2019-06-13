#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

const int HcalZDCDetId::kZDCChannelMask;
const int HcalZDCDetId::kZDCSectionMask;
const int HcalZDCDetId::kZDCSectionOffset;
const int HcalZDCDetId::kZDCZsideMask;
const int HcalZDCDetId::kZDCRPDMask;
const int HcalZDCDetId::SubdetectorId;

HcalZDCDetId::HcalZDCDetId() : DetId() {}

HcalZDCDetId::HcalZDCDetId(uint32_t rawid) : DetId(rawid) {}

HcalZDCDetId::HcalZDCDetId(Section section, bool true_for_positive_eta, int channel)
    : DetId(DetId::Calo, SubdetectorId) {
  if (section == RPD) {
    id_ |= (Unknown & kZDCSectionMask) << kZDCSectionOffset;
    id_ |= kZDCRPDMask;
    id_ |= ((channel - 1) & kZDCChannelMask);
  } else {
    id_ |= (section & kZDCSectionMask) << kZDCSectionOffset;
    id_ |= (channel & kZDCChannelMask);
  }
  if (true_for_positive_eta)
    id_ |= kZDCZsideMask;
}

HcalZDCDetId::HcalZDCDetId(const DetId& gen) {
  if (!gen.null() && (gen.det() != Calo || gen.subdetId() != SubdetectorId)) {
    throw cms::Exception("Invalid DetId") << "Cannot initialize ZDCDetId from " << std::hex << gen.rawId() << std::dec;
  }
  id_ = gen.rawId();
}

HcalZDCDetId& HcalZDCDetId::operator=(const DetId& gen) {
  if (!gen.null() && (gen.det() != Calo || gen.subdetId() != SubdetectorId)) {
    throw cms::Exception("Invalid DetId") << "Cannot assign ZDCDetId from " << std::hex << gen.rawId() << std::dec;
  }
  id_ = gen.rawId();
  return *this;
}

HcalZDCDetId::Section HcalZDCDetId::section() const {
  if (id_ & kZDCRPDMask)
    return RPD;
  else
    return (Section)((id_ >> kZDCSectionOffset) & kZDCSectionMask);
}

int HcalZDCDetId::depth() const {
  const int se(section());
  if (se == EM)
    return 1;
  else if (se == HAD)
    return (channel() + 2);
  else if (se == RPD)
    return 2;
  else
    return channel();
}

int HcalZDCDetId::channel() const {
  const int se(section());
  if (se == RPD)
    return (1 + (id_ & kZDCChannelMask));
  else
    return (id_ & kZDCChannelMask);
}

uint32_t HcalZDCDetId::denseIndex() const {
  const int se(section());
  uint32_t di =
      (channel() - 1 +
       (se == RPD ? 2 * kDepRun1 + (zside() < 0 ? 0 : kDepRPD)
                  : ((zside() < 0 ? 0 : kDepRun1) + (se == HAD ? kDepEM : (se == LUM ? kDepEM + kDepHAD : 0)))));
  return di;
}

HcalZDCDetId HcalZDCDetId::detIdFromDenseIndex(uint32_t di) {
  if (validDenseIndex(di)) {
    bool lz(false);
    uint32_t dp(0);
    Section se(Unknown);
    if (di >= 2 * kDepRun1) {
      lz = (di >= (kDepRun1 + kDepTot));
      se = RPD;
      dp = 1 + ((di - 2 * kDepRun1) % kDepRPD);
    } else {
      lz = (di >= kDepRun1);
      uint32_t in = (di % kDepRun1);
      se = (in < kDepEM ? EM : (in < kDepEM + kDepHAD ? HAD : LUM));
      dp = (EM == se ? in + 1 : (HAD == se ? in - kDepEM + 1 : in - kDepEM - kDepHAD + 1));
    }
    return HcalZDCDetId(se, lz, dp);
  } else {
    return HcalZDCDetId();
  }
}

bool HcalZDCDetId::validDetId(Section se, int dp) {
  bool flag = (dp >= 1 && (((se == EM) && (dp <= kDepEM)) || ((se == HAD) && (dp <= kDepHAD)) ||
                           ((se == LUM) && (dp <= kDepLUM)) || ((se == RPD) && (dp <= kDepRPD))));
  return flag;
}

std::ostream& operator<<(std::ostream& s, const HcalZDCDetId& id) {
  s << "(ZDC" << ((id.zside() == 1) ? ("+") : ("-"));
  switch (id.section()) {
    case (HcalZDCDetId::EM):
      s << " EM ";
      break;
    case (HcalZDCDetId::HAD):
      s << " HAD ";
      break;
    case (HcalZDCDetId::LUM):
      s << " LUM ";
      break;
    case (HcalZDCDetId::RPD):
      s << " RPD ";
      break;
    default:
      s << " UNKNOWN ";
  }
  return s << id.channel() << "," << id.depth() << ')';
}
