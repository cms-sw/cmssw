#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include <iomanip>

std::ostream& operator<<(std::ostream& os, const ETLDetId& id) {
  os << (MTDDetId&)id;
  os << " ETL " << std::endl
     << " Side        : " << id.mtdSide() << std::endl
     << " Ring        : " << id.mtdRR() << "    "
     << " Disc/Side/Sector = " << id.nDisc() << " " << id.discSide() << " " << id.sector() << std::endl
     << " Module      : " << id.module() << std::endl
     << " Module type : " << id.modType() << std::endl;
  return os;
}

uint32_t ETLDetId::newForm(const uint32_t& rawid) {
  uint32_t rawid_new = 0;
  rawid_new |= (MTDType::ETL & ETLDetId::kMTDsubdMask) << ETLDetId::kMTDsubdOffset |
               (mtdSide() & ETLDetId::kZsideMask) << ETLDetId::kZsideOffset |
               (mtdRR() & ETLDetId::kRodRingMask) << ETLDetId::kRodRingOffset |
               (module() & ETLDetId::kETLmoduleMask) << (ETLDetId::kETLmoduleOffset - 2) |
               (modType() & ETLDetId::kETLmodTypeMask) << (kETLmodTypeOffset - 2);
  return rawid_new;
}
