#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include <iostream>

void ChamberLocationSpec::print( int depth ) const
{
  if (depth <0) return;
  std::cout << " ChamberLocationSpec: " << std::endl
            << " --->DiskOrWheel: " << diskOrWheel
            << " Layer: " << layer
            << " Sector: " << sector
            << " Subsector: " << subsector
            << " ChamberLocationName: " << chamberLocationName
            << " FebZOrnt: " << febZOrnt
            << " FebZRadOrnt: " << febZRadOrnt
            << " BarrelOrEndcap: " << barrelOrEndcap
            << std::endl; 
}
