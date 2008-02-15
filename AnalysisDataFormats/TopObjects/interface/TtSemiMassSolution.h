#ifndef TopObjects_TtSemiMassSolution_h
#define TopObjects_TtSemiMassSolution_h

#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TtSemiEvtSolution.h"

class TtSemiMassSolution : public TtSemiEvtSolution
{
 public:
  
  TtSemiMassSolution();
  TtSemiMassSolution(TtSemiEvtSolution);
  virtual ~TtSemiMassSolution();
  
  void setMtopUncertainty(double dm) { dmtop_ = dm; };
  void setScanValues(std::vector<std::pair<double,double> >);
  
  double getMtopUncertainty() const { return dmtop_; };
  std::vector<std::pair<double,double> > getScanValues() const { return scanValues; };
  
 private:
  
  double dmtop_;
  std::vector<std::pair<double,double> > scanValues;
};

#endif
