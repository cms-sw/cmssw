#ifndef OSCAR_UnitConverter_h
#define OSCAR_UnitConverter_h

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"

#include <iostream>
#include <string>
// inserts a multiplication '*' between a value and a unit
// returned from G4BestUnit
class CocoaBestUnit;
class UnitConverter;

//ostream & operator<<(ostream &, const UnitConverter & );

class UnitConverter {
public:
  UnitConverter(ALIdouble val, const ALIstring& category);
  ~UnitConverter();
  std::string ucstring();
  //friend ostream& operator(std::ostream & ,const UnitConverter & VU);

  CocoaBestUnit* bu_;
  bool angl_;
};

#endif
