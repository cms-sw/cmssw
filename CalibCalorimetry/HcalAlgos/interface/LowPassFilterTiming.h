#ifndef CalibCalorimetry_HcalAlgos_LowPassFilterTiming_h_
#define CalibCalorimetry_HcalAlgos_LowPassFilterTiming_h_

class LowPassFilterTiming {
public:
  unsigned nParameters() const;

  double operator()(double currentIn, const double* params, unsigned nParams) const;
};

#endif  // CalibCalorimetry_HcalAlgos_LowPassFilterTiming_h_
