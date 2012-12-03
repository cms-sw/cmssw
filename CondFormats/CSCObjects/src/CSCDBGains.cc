#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include <iostream>

std::ostream & operator<<(std::ostream & os, const CSCDBGains & cscdb)
{
  for ( size_t i = 0; i < cscdb.gains.size(); ++i )
  {
    os <<  "elem: " << i << " gain_slope: " << cscdb.gains[i].gain_slope << "\n";
  }
  return os;
}
