/** \file
 * Impl of GEMDetId
 */

#include "DataFormats/MuonDetId/interface/GEMDetId.h"

GEMSubDetId::Station GEMDetId::subsystem() const { return GEMSubDetId::station(station()); }

bool GEMDetId::isGE11() const { return subsystem() == GEMSubDetId::Station::GE11; }

bool GEMDetId::isGE21() const { return subsystem() == GEMSubDetId::Station::GE21; }

bool GEMDetId::isME0() const { return subsystem() == GEMSubDetId::Station::ME0; }

std::ostream& operator<<(std::ostream& os, const GEMDetId& id) {
  os << " Re " << id.region() << " Ri " << id.ring() << " St " << id.station() << " La " << id.layer() << " Ch "
     << id.chamber() << " Ro " << id.roll() << " ";

  return os;
}
