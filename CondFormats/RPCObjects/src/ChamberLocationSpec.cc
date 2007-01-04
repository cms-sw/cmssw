#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include <sstream>

std::string ChamberLocationSpec::print( int depth ) const
{
  std::ostringstream str;
  if (depth >=0) {
    str  << " ChamberLocationSpec: " << std::endl
         << " --->DiskOrWheel: " << diskOrWheel
         << " Layer: " << layer
         << " Sector: " << sector
         << " Subsector: " << subsector
         << " ChamberLocationName: " << chamberLocationName
         << " FebZOrnt: " << febZOrnt
         << " FebZRadOrnt: " << febZRadOrnt
         << " BarrelOrEndcap: " << barrelOrEndcap;
  }
  return str.str();
}
