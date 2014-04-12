#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include <iostream>

std::ostream & operator<<(std::ostream & os, const CSCDBCrosstalk & cscdb)
{
  for ( size_t i = 0; i < cscdb.crosstalk.size(); ++i )
  {
    os <<  "elem: " << i <<
    " R slope: "   << cscdb.crosstalk[i].xtalk_slope_right     <<
    " intercept: " << cscdb.crosstalk[i].xtalk_intercept_right <<
    " L slope: "   << cscdb.crosstalk[i].xtalk_slope_left      <<
    " intercept: " << cscdb.crosstalk[i].xtalk_intercept_left  << "\n";
  }
  return os;
}
