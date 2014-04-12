

#include "DetectorDescription/Base/interface/DDTypes.h"
#include <iostream>
////////// output operator for printing the arguments of an algorithm

std::ostream & operator<<(std::ostream & os, const DDNumericArguments & t)
{
  DDNumericArguments::const_iterator it(t.begin()), ed(t.end());
  for(; it != ed; ++it) {
    os << it->first << '=' << it->second << std::endl;
  }
  return os;
}

std::ostream & operator<<(std::ostream & os, const DDStringArguments & t)
{
  DDStringArguments::const_iterator it(t.begin()), ed(t.end());
  for(; it != ed; ++it) {
    os << it->first << '=' << it->second << std::endl;
  }
  return os;
}

std::ostream & operator<<(std::ostream & os, const DDVectorArguments & t)
{
  DDVectorArguments::const_iterator it(t.begin()), ed(t.end());
  for(; it != ed; ++it) {
    os << it->first << ": ";
    std::vector<double>::const_iterator vit(it->second.begin()), ved(it->second.end());
    for(;vit!=ved;++vit) {
      os << *vit << ' ';
    }
    os << std::endl;
  }
  return os;
}

std::ostream & operator<<(std::ostream & os, const DDMapArguments & t)
{
  DDMapArguments::const_iterator it(t.begin()), ed(t.end());
  for(; it != ed; ++it) {
    os << it->first << ": ";
    std::map<std::string,double>::const_iterator mit(it->second.begin()), med(it->second.end());
    for(;mit!=med;++mit) {
      os << mit->first << '=' << mit->second << ' ';
    }
    os << std::endl;
  }
  return os;
}

std::ostream & operator<<(std::ostream & os, const DDStringVectorArguments & t)
{
  DDStringVectorArguments::const_iterator it(t.begin()), ed(t.end());
  for(; it != ed; ++it) {
    os << it->first << ": ";
    std::vector<std::string>::const_iterator vit(it->second.begin()), ved(it->second.end());
    for(; vit!=ved; ++vit) {
     os << *vit << ' '; 
    }
    os << std::endl;
  }
  return os;
}

