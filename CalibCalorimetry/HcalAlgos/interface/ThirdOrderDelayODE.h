#ifndef CalibCalorimetry_HcalAlgos_ThirdOrderDelayODE_h_
#define CalibCalorimetry_HcalAlgos_ThirdOrderDelayODE_h_

//
// Equation a/6*tau^3*V_out''' + b/2*tau^2*V_out'' + c*tau*V_out' + V_out = V_in,
// with parameters "a", "b", and "c". a = 1, b = 1, c = 1 corresponds to the
// Pade table delay equation with row = 0 and column = 3.
//
class ThirdOrderDelayODE {
public:
  inline ThirdOrderDelayODE(unsigned /* r */, unsigned /* c */) : a_(1.0) {}

  void calculate(double tau,
                 double inputCurrent,
                 double dIdt,
                 double d2Id2t,
                 const double* x,
                 unsigned lenX,
                 unsigned firstNode,
                 double* derivative) const;

  inline unsigned getPadeRow() const { return 0U; }
  inline unsigned getPadeColumn() const { return 3U; }
  inline unsigned nParameters() const { return 3U; }

  // The parameters should be set to the logs of their actual values
  void setParameters(const double* pars, unsigned nPars);

private:
  double a_;
  double b_;
  double c_;
};

#endif  // CalibCalorimetry_HcalAlgos_ThirdOrderDelayODE_h_
