#ifndef GlobalTrigger_L1GlobalTriggerFunctions_h
#define GlobalTrigger_L1GlobalTriggerFunctions_h

#include <string>
#include <vector>

#include <fstream>
#include <iostream>
#include <sstream>

/// factorial function
int factorial(int n);

/// convert a hex string to a vector of 64-bit integers,
/// with the vector size depending on the string size
/// return false in case of error
bool hexStringToInt64(const std::string &, std::vector<unsigned long long> &);

/// convert a string to a integer-type number
/// the third parameter of stringToNumber should be
/// one of std::hex, std::dec or std::oct
template <class T>
bool stringToNumber(T &tmpl, const std::string &str, std::ios_base &(*f)(std::ios_base &)) {
  std::istringstream iss(str);
  return !(iss >> f >> tmpl).fail();
}

#endif /*GlobalTrigger_L1GlobalTriggerFunctions_h*/
