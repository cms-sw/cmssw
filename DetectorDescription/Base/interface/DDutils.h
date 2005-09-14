#ifndef DDutils_hh
#define DDutils_hh 1

#include <string>
#include <vector>
#include <iostream>

std::vector<int>  dbl_to_int ( const std::vector<double> & vecdbl);
//Converts a std::vector of doubles to a std::vector of int

std::string dbl_to_string (const double& in );
std::string int_to_string (const int& in );
//Converts int or double to std::string

#endif
