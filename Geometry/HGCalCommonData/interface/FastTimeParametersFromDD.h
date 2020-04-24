#ifndef HGCalCommonData_FastTimeParametersFromDD_h
#define HGCalCommonData_FastTimeParametersFromDD_h

#include <string>
#include "DetectorDescription/Core/interface/DDsvalues.h"

class DDCompactView;
class FastTimeParameters;

class FastTimeParametersFromDD {
public:
  FastTimeParametersFromDD() {}
  virtual ~FastTimeParametersFromDD() {}

  bool build(const DDCompactView*,  FastTimeParameters&, const std::string&,
	     const int);
private:
  std::vector<double> getDDDArray(const std::string &, const DDsvalues_type &);
};

#endif
