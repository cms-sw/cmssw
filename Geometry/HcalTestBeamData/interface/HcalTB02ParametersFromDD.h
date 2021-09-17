#ifndef HcalTestBeamData_HcalTB02ParametersFromDD_h
#define HcalTestBeamData_HcalTB02ParametersFromDD_h

#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/HcalTestBeamData/interface/HcalTB02Parameters.h"

class HcalTB02ParametersFromDD {
public:
  HcalTB02ParametersFromDD() = default;

  bool build(const DDCompactView* cpv, HcalTB02Parameters& php, const std::string& name);
  bool build(const cms::DDCompactView* cpv, HcalTB02Parameters& php, const std::string& name);

private:
  static constexpr double k_ScaleFromDDDToG4 = 1.0;
  static constexpr double k_ScaleFromDD4HepToG4 = 10.0;
};

#endif
