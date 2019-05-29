#ifndef DDValuePair_h
#define DDValuePair_h

#include <string>
#include <map>
//#include <utility>

struct DDValuePair : public std::pair<std::string, double> {
  DDValuePair() {}
  DDValuePair(const std::string& s, double d) : std::pair<std::string, double>(s, d) {}
  DDValuePair(const std::string& s) : std::pair<std::string, double>(s, 0) {}
  DDValuePair(double d) : std::pair<std::string, double>("", d) {}

  operator const std::string&() const { return first; }
  operator std::string&() { return first; }
  operator const double&() const { return second; }
  operator double&() { return second; }
};

std::ostream& operator<<(std::ostream& o, const DDValuePair& v);

#endif
