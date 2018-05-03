#ifndef HGCalCommonData_HGCalParametersFromDD_h
#define HGCalCommonData_HGCalParametersFromDD_h

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include <string>
#include <vector>

class DDCompactView;
class HGCalParameters;

class HGCalParametersFromDD {
public:
  HGCalParametersFromDD() {}
  virtual ~HGCalParametersFromDD() {}

  bool build(const DDCompactView*,  HGCalParameters&, const std::string&,
	     const std::string&, const std::string&);

private:
  void                getCellPosition(HGCalParameters& php, int type);
  double              getDDDValue(const char* s, const DDsvalues_type& sv);
  std::vector<double> getDDDArray(const char* s, const DDsvalues_type& sv);
};

#endif
