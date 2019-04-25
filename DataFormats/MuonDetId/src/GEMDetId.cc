/** \file
 * Impl of GEMDetId
 */

#include "DataFormats/MuonDetId/interface/GEMDetId.h"

std::ostream& operator<<( std::ostream& os, const GEMDetId& id ){

  os << " Region "  << id.region()
     << " Ring "    << id.ring()
     << " Station " << id.station()
     << " Layer "   << id.layer()
     << " Chamber " << id.chamber()
     << " Roll "    << id.roll()
     <<" ";

  return os;
}


