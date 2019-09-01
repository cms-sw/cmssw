#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

HcalCastorDetId::HcalCastorDetId() : DetId() {}

HcalCastorDetId::HcalCastorDetId(uint32_t rawid) : DetId(rawid) {}

void HcalCastorDetId::buildMe(Section section, bool true_for_positive_eta, int sector, int module) {
  sector -= 1;  // we count sector from 1-16 instead of 0-15
  id_ |= (true_for_positive_eta << 8) | (sector << 4) | module;
}

HcalCastorDetId::HcalCastorDetId(Section section, bool true_for_positive_eta, int sector, int module)
    : DetId(DetId::Calo, SubdetectorId) {
  buildMe(section, true_for_positive_eta, sector, module);
}

HcalCastorDetId::HcalCastorDetId(bool true_for_positive_eta, int sector, int module)
    : DetId(DetId::Calo, SubdetectorId) {
  buildMe(Section(Unknown), true_for_positive_eta, sector, module);
}

HcalCastorDetId::HcalCastorDetId(const DetId& gen) {
  if (!gen.null() && (gen.det() != DetId::Calo || gen.subdetId() != SubdetectorId)) {
    throw cms::Exception("Invalid DetId")
        << "Cannot initialize CASTORDetId from " << std::hex << gen.rawId() << std::dec;
  }
  id_ = gen.rawId();
}

HcalCastorDetId& HcalCastorDetId::operator=(const DetId& gen) {
  if (!gen.null() && (gen.det() != DetId::Calo || gen.subdetId() != SubdetectorId)) {
    throw cms::Exception("Invalid DetId") << "Cannot assign Castor DetId from " << std::hex << gen.rawId() << std::dec;
  }

  id_ = gen.rawId();

  return *this;
}

/*
int HcalCastorDetId::channel() const {
  int channelid = 16*(sector-1)+module;
  return channelid;
}
*/

HcalCastorDetId::Section HcalCastorDetId::section() const {
  const int mod = module();

  Section sect;
  if (mod <= 2) {
    sect = HcalCastorDetId::EM;
  } else {
    if (mod > 2 && mod <= 14) {
      sect = HcalCastorDetId::HAD;
    } else {
      sect = HcalCastorDetId::Unknown;
    }
  }
  return sect;
}

uint32_t HcalCastorDetId::denseIndex() const {
  return (kNumberCellsPerEnd * (zside() + 1) / 2 + kNumberSectorsPerEnd * (module() - 1) + sector() - 1);
}

bool HcalCastorDetId::validDetId(Section iSection, bool posEta, int iSector, int iModule) {
  return (0 < iSector && kNumberSectorsPerEnd >= iSector && 0 < iModule && kNumberModulesPerEnd >= iModule);
}

HcalCastorDetId HcalCastorDetId::detIdFromDenseIndex(uint32_t di) {
  return HcalCastorDetId(
      kNumberCellsPerEnd <= di, di % kNumberSectorsPerEnd + 1, (di % kNumberCellsPerEnd) / kNumberSectorsPerEnd + 1);
}

std::ostream& operator<<(std::ostream& s, const HcalCastorDetId& id) {
  s << "(CASTOR" << ((id.zside() == 1) ? ("+") : ("-"));

  switch (id.section()) {
    case (HcalCastorDetId::EM):
      s << " EM ";
      break;
    case (HcalCastorDetId::HAD):
      s << " HAD ";
      break;
    default:
      s << " UNKNOWN ";
  }

  return s << id.sector() << ',' << id.module() << ',' << ')';
}
