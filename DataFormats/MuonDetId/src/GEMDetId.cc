/** \file
 * Impl of GEMDetId
 */

#include "DataFormats/MuonDetId/interface/GEMDetId.h"

std::ostream& operator<<(std::ostream& os, const GEMDetId& id) {
  os << " Re " << id.region() << " Ri " << id.ring() << " St " << id.station() << " La " << id.layer() << " Ch "
     << id.chamber() << " Ro " << id.roll() << " ";

  return os;
}
