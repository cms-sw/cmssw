#ifndef HGCalCommonData_HGCalParametersFromDD_h
#define HGCalCommonData_HGCalParametersFromDD_h

#include <string>

class DDCompactView;
class HGCalParameters;

class HGCalParametersFromDD {
public:
  HGCalParametersFromDD() {}
  virtual ~HGCalParametersFromDD() {}

  bool build(const DDCompactView*,  HGCalParameters&, const std::string&,
	     const std::string&, const std::string&);
};

#endif
