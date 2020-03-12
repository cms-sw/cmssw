#ifndef HcalCommonData_HcalSimParametersFromDD_h
#define HcalCommonData_HcalSimParametersFromDD_h

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include <string>

class HcalSimulationParameters;

class HcalSimParametersFromDD {
public:
  HcalSimParametersFromDD() = default;

  bool build(const DDCompactView*, HcalSimulationParameters&);
  bool build(const cms::DDCompactView*, HcalSimulationParameters&);

private:
  bool buildParameters(const HcalSimulationParameters&);
  void fillNameVector(const DDCompactView*, const std::string&, const std::string&, std::vector<std::string>&);
  void fillNameVector(const cms::DDCompactView*, const std::string&, std::vector<std::string>&);
  void fillPMTs(const std::vector<double>&, bool, HcalSimulationParameters&);
  bool isItHF(const std::string&, const HcalSimulationParameters&);
  std::vector<std::string> getNames(DDFilteredView& fv);
  std::vector<std::string> getNames(cms::DDFilteredView& fv);
  std::vector<double> getDDDArray(const std::string& str, const DDsvalues_type& sv, int& nmin);
};

#endif
