#ifndef HGCalCommonData_HGCalParametersFromDD_h
#define HGCalCommonData_HGCalParametersFromDD_h

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include <string>

class DDCompactView;
class HGCalParameters;

class HGCalParametersFromDD {
public:
  HGCalParametersFromDD() {}
  virtual ~HGCalParametersFromDD() {}

  bool build(const DDCompactView*,  HGCalParameters&, const std::string&,
	     const std::string&, const std::string&);

private:
  double getDDDValue(const char* s, const DDsvalues_type& sv);
};

#endif
