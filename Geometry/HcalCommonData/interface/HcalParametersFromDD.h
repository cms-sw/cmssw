#ifndef HcalCommonData_HcalParametersFromDD_h
#define HcalCommonData_HcalParametersFromDD_h

class DDCompactView;
class HcalParameters;

class HcalParametersFromDD {
public:
  HcalParametersFromDD() {}
  virtual ~HcalParametersFromDD() {}

  bool build(const DDCompactView*,  HcalParameters& );
};

#endif
