#ifndef EcalCommonData_EcalSimParametersFromDD_h
#define EcalCommonData_EcalSimParametersFromDD_h

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <vector>

class DDFilteredView;
class EcalSimulationParameters;

class EcalSimParametersFromDD {
public:
  EcalSimParametersFromDD() = default;

  bool build(const DDCompactView*, const std::string& name, EcalSimulationParameters&);
  bool build(const cms::DDCompactView*, const std::string& name, EcalSimulationParameters&);

private:
  bool buildParameters(const EcalSimulationParameters&);
  std::vector<std::string> getStringArray(const std::string&, const DDsvalues_type&);
  std::vector<double> getDDDArray(const std::string&, const DDsvalues_type&);
};

#endif
