#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

std::ostream& operator<<(std::ostream& s, const HcalZDCDetId& id) {
  s << "(Det " << id.det() << ":" << DetId::Calo << " subdet " << id.subdetId() << ":" << HcalZDCDetId::SubdetectorId
    << " ZDC" << ((id.zside() == 1) ? ("+") : ("-"));
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
