#include "CondFormats/CSCObjects/interface/CSCDBGasGainCorrection.h"
#include <iostream>

std::ostream & operator<<(std::ostream & os, const CSCDBGasGainCorrection & cscdb)
{
  for ( size_t i = 0; i < cscdb.gasGainCorr.size(); ++i )
  {
    os <<  "elem: " << i << " gas gain corr: " << cscdb.gasGainCorr[i].gainCorr << "\n";
  }
  return os;
}
