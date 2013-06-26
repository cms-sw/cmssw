#include "CondFormats/CSCObjects/interface/CSCDBChipSpeedCorrection.h"
#include <iostream>

std::ostream & operator<<(std::ostream & os, const CSCDBChipSpeedCorrection & cscdb)
{
  for ( size_t i = 0; i < cscdb.chipSpeedCorr.size(); ++i )
  {
    os <<  "elem: " << i << " chip speed corr: " << cscdb.chipSpeedCorr[i].speedCorr << "\n";
  }
  return os;
}
