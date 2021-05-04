#ifndef HcalTestBeamData_HcalTB06BeamParametersFromDD_h
#define HcalTestBeamData_HcalTB06BeamParametersFromDD_h

#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/HcalTestBeamData/interface/HcalTB06BeamParameters.h"

class HcalTB06BeamParametersFromDD {
public:
  HcalTB06BeamParametersFromDD() = default;

  bool build(const DDCompactView* cpv, HcalTB06BeamParameters& php, const std::string& name1, const std::string& name2);
  bool build(const cms::DDCompactView* cpv,
             HcalTB06BeamParameters& php,
             const std::string& name1,
             const std::string& name2);

private:
  bool build(HcalTB06BeamParameters& php,
             const std::vector<std::string>& matNames,
             const std::vector<int>& nocc,
             const std::string& name1,
             const std::string& name2);
  std::vector<std::string> getNames(DDFilteredView& fv);
  std::vector<std::string> getNames(cms::DDFilteredView& fv);
};

#endif
