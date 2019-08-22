#ifndef HcalCommonData_HcalSimParametersFromDD_h
#define HcalCommonData_HcalSimParametersFromDD_h

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include <string>

class DDCompactView;
class HcalSimulationParameters;

class HcalSimParametersFromDD {
public:
  HcalSimParametersFromDD() {}
  virtual ~HcalSimParametersFromDD() {}

  bool build(const DDCompactView*, HcalSimulationParameters&);

private:
  std::vector<double> getDDDArray(const std::string& str, const DDsvalues_type& sv, int& nmin);
};

#endif
