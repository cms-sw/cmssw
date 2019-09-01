#ifndef HcalCommonData_HcalSimParametersFromDD_h
#define HcalCommonData_HcalSimParametersFromDD_h

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include <string>

class DDCompactView;
class DDFilteredView;
class HcalSimulationParameters;

class HcalSimParametersFromDD {
public:
  HcalSimParametersFromDD() {}
  virtual ~HcalSimParametersFromDD() {}

  bool build(const DDCompactView*, HcalSimulationParameters&);

private:
  void fillNameVector(const DDCompactView*, const std::string&, const std::string&, std::vector<std::string>&);
  bool isItHF(const std::string&, const HcalSimulationParameters&);
  std::vector<std::string> getNames(DDFilteredView& fv);
  std::vector<double> getDDDArray(const std::string& str, const DDsvalues_type& sv, int& nmin);
};

#endif
