#ifndef CalibCalorimetry_HcalAlgos_PadeTableODE_h_
#define CalibCalorimetry_HcalAlgos_PadeTableODE_h_

//
// Differential equations are built using the delay formula
// I_out(s) = I_in(s) exp(-tau s), where I_out(s), etc. are the Laplace
// transforms. exp(-tau s) is then represented by fractions according
// to the Pade table. See http://en.wikipedia.org/wiki/Pade_table and
// replace z by (-tau s).
//
class PadeTableODE {
public:
  PadeTableODE(unsigned padeRow, unsigned padeColumn);

  void calculate(double tau,
                 double inputCurrent,
                 double dIdt,
                 double d2Id2t,
                 const double* x,
                 unsigned lenX,
                 unsigned firstNode,
                 double* derivative) const;

  inline unsigned getPadeRow() const { return row_; }
  inline unsigned getPadeColumn() const { return col_; }
  inline unsigned nParameters() const { return 0U; }
  void setParameters(const double* pars, unsigned nPars);

private:
  unsigned row_;
  unsigned col_;
};

#endif  // CalibCalorimetry_HcalAlgos_PadeTableODE_h_
