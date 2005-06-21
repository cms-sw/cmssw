#ifndef DDValuePair_h
#define DDValuePair_h

#include <string>
#include <utility>
using std::string;
using std::pair;

struct DDValuePair : public pair<string,double>
{
  DDValuePair() { }
  DDValuePair(const string & s, double d) : pair<string,double>(s,d) { }
  DDValuePair(const string & s) : pair<string,double>(s,0) { }
  DDValuePair(double d) : pair<string,double>("",d) { }
  
  operator const string&() const { return first; }
  operator string&() { return first; }
  operator const double&() const { return second; }
  operator double&() { return second; }
  
};

#endif
