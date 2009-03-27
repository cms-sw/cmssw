#include "DetectorDescription/Base/interface/DDutils.h"
#include <vector>
#include <string>
#include <sstream>

std::vector<int> dbl_to_int ( const std::vector<double> & vecdbl) {
  std::vector<int> tInt;
  for (std::vector<double>::const_iterator vd_it = vecdbl.begin(); vd_it != vecdbl.end(); ++vd_it) {
    tInt.push_back(int(*vd_it));
  }
  return tInt;
}

std::string dbl_to_string (const double& in) {
  return int_to_string( int (in) );
}

std::string int_to_string(const int& in) {

  std::ostringstream ostr;
  ostr << in;
  return ostr.str();

}
