#ifndef DDutils_hh
#define DDutils_hh 1

#include <string>
#include <vector>

/// Converts a std::vector of doubles to a std::vector of int
inline std::vector<int> dbl_to_int( const std::vector<double> & vecdbl ) {
  std::vector<int> tInt( vecdbl.begin(), vecdbl.end());
  return tInt;
}

#endif
