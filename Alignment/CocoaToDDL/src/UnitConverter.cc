#include "Alignment/CocoaToDDL/interface/UnitConverter.h"
#include "Alignment/CocoaToDDL/interface/CocoaUnitsTable.h"
#include <CLHEP/Units/SystemOfUnits.h>
#include <sstream>

/*
UnitConverter::UnitConverter(const G4BestUnit & bu)
 : bu_(bu)
{
  //ostrstream s;
  //s << bu;
  
}
*/

UnitConverter::UnitConverter(ALIdouble val, const ALIstring& category)
    : bu_(new CocoaBestUnit(val, category)), angl_(false) {
  if (category == "Angle")
    angl_ = true;
}

UnitConverter::~UnitConverter() { delete bu_; }

std::string UnitConverter::ucstring() {
  std::ostringstream str;

  if (angl_) {
    str.precision(11);
    double x = (*(bu_->GetValue())) / CLHEP::deg;
    str << x << std::string("*deg") << '\0';
    return std::string(str.str());

  } else {
    str << *bu_ << '\0';
    std::string s(str.str());
    return s.replace(s.find(' '), 1, "*");
  }
  //return s;
}

/*
ostream & operator<<(ostream & os, const UnitConverter & uc)
{
  std::ostringstream temp;
  //temp << uc.bu_;
  //temp << '\0';
  //string s(temp.str());
  //cout << "NOW: " << s << endl;
  os << *(uc.bu_);
}

*/
