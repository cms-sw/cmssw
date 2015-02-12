#ifndef HcalTowerAlgo_HcalParametersFromDD_h
#define HcalTowerAlgo_HcalParametersFromDD_h

class DDCompactView;
class PHcalParameters;

class HcalParametersFromDD {
 public:
  HcalParametersFromDD() {}
  virtual ~HcalParametersFromDD() {}

  bool build( const DDCompactView*,
	      PHcalParameters& );
};

#endif
