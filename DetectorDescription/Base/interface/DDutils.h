#ifndef DDutils_hh
#define DDutils_hh 1

#include <string>
#include <vector>
#include <iostream>

/// Converts a std::vector of doubles to a std::vector of int
std::vector<int>  dbl_to_int ( const std::vector<double> & vecdbl);

/// Converts only the integer part of a double to a string.
std::string dbl_to_string (const double& in );

//Converts int or double to std::string
std::string int_to_string (const int& in );

#endif
