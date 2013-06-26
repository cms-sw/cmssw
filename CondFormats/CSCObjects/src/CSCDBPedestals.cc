#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include <iostream>

std::ostream & operator<<(std::ostream & os, const CSCDBPedestals & cscdb)
{
  for ( size_t i = 0; i < cscdb.pedestals.size(); ++i )
  {
    os <<  "elem: " << i << " pedestal: " << cscdb.pedestals[i].ped << " rms: " << cscdb.pedestals[i].rms << "\n";
  }
  return os;
}
