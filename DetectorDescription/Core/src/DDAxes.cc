#include "DetectorDescription/Core/interface/DDAxes.h"

namespace DDI { } using namespace DDI;

namespace {
  template <class T>
  struct entry {                                                                                                                                                    
     char const* label;
     T           value;
  };

  constexpr bool same(char const *x, char const *y) {
    return !*x && !*y     ? true                                                                                                                                
           : /* default */    (*x == *y && same(x+1, y+1));                                                                                                       
  }

  template <class T>
  constexpr T keyToValue(char const *label, entry<T> const *entries) {                                                                                  
     return !entries->label ? entries->value                         
            : same(entries->label, label) ? entries->value
            : /*default*/                   keyToValue(label, entries+1);                                                                                 
  }

  template <class T>
  constexpr char const*valueToKey(T value, entry<T> const *entries) {
     return !entries->label ? entries->label
            : entries->value == value ? entries->label
            : /*default*/       valueToKey(value, entries+1);
  }

  entry<DDAxes> axesNames[] = {
    {"x", x},
    {"y", y},
    {"z", z},
    {"rho", rho},
    {"radial3D", radial3D},
    {"phi", phi},
    {"undefined", undefined},
    {0, undefined},
  };
}

char const*
DDAxesNames::name(const DDAxes& s) 
{
  return valueToKey(s, axesNames);
}

DDAxes
DDAxesNames::index(const std::string & s)
{
  return keyToValue(s.c_str(), axesNames);
}
