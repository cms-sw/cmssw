#ifndef HcalCommonData_CaloSimParametersFromDD_h
#define HcalCommonData_CaloSimParametersFromDD_h

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include <string>

class DDCompactView;
class DDFilteredView;
class CaloSimulationParameters;

class CaloSimParametersFromDD {
public:
  CaloSimParametersFromDD(bool fromDD4Hep);
  virtual ~CaloSimParametersFromDD() {}

  bool build(const DDCompactView*, CaloSimulationParameters&);

private:
  std::vector<std::string> getNames(const std::string&, const DDsvalues_type&, bool);
  std::vector<int> getNumbers(const std::string&, const DDsvalues_type&, bool);

  bool fromDD4Hep_;
};

#endif
