#ifndef _COMP2REFCHI2_ROOT_H
#define _COMP2REFCHI2_ROOT_H

#include "DQMServices/Core/interface/QualTestBase.h"

// comparison to reference using the ROOT chi^2 algorithm
class Comp2RefChi2ROOT : public Comp2RefBase<TH1F>
{
 public:
    
  Comp2RefChi2ROOT(void) : Comp2RefBase<TH1F>(){}
  virtual ~Comp2RefChi2ROOT(void){}
  // run the test (result: [0, 1] or <0 for failure)
  float runTest(const TH1F * const h);
  //get  algorithm name
  static std::string getAlgoName(void){return "Comp2RefChi2";}
  // true if test cannot run
  bool isInvalid(const TH1F * const h);
 protected:
  // # of degrees of freedom and chi^2 for test
  int Ndof_;
  float chi2_;

  void resetResults(void);
  // # of bins for test & reference histogram 
  Int_t nbins1;
  Int_t nbins2;
};

#endif
