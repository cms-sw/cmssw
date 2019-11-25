#ifndef HcalCommonData_HcalParametersFromDD_h
#define HcalCommonData_HcalParametersFromDD_h

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

class HcalParameters;

class HcalParametersFromDD {
public:
  HcalParametersFromDD() = default;

  bool build(const DDCompactView*, HcalParameters&);
  bool build(const cms::DDCompactView*, HcalParameters&) { return true; }

private:
  bool build(const HcalParameters&);
};

#endif
