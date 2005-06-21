#include "DetectorDescription/DDBase/interface/DDutils.h"
#include <fstream>
#include <cmath>

std::vector<int> dbl_to_int ( const std::vector<double> & vecdbl) {
  std::vector<int> tInt;
  for (std::vector<double>::const_iterator vd_it = vecdbl.begin(); vd_it != vecdbl.end(); vd_it++) {
    tInt.push_back(int(*vd_it));
  }
  return tInt;
}

std::string dbl_to_string (const double& in) {
  return int_to_string( int (in) );
}

std::string int_to_string(const int& in) {
  if (in < 0) return std::string("-") + int_to_string(in * -1);
  
  if (in < 10) {
    switch (in) {
    case 0: 
      return std::string("0");
      break;

    case 1:
      return std::string("1");
      break;

    case 2:
      return std::string("2");
      break;

    case 3:
      return std::string("3");
      break;

    case 4:
      return std::string("4");
      break;

    case 5:
      return std::string("5");
      break;

    case 6:
      return std::string("6");
      break;

    case 7:
      return std::string("7");
      break;

    case 8:
      return std::string("8");
      break;
      
    case 9:
      return std::string("9");
      break;

    default:
      return std::string(" ");
    }
  }
  return int_to_string(in / 10) + int_to_string(in % 10);
}
