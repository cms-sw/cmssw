#ifndef _COMP2REFKOLMOGOROV_ROOT_H
#define _COMP2REFKOLMOGOROV_ROOT_H

#include "DQMServices/Core/interface/QualTestBase.h"

/// comparison to reference using the ROOT Kolmogorov algorithm
class Comp2RefKolmogorovROOT : public Comp2RefBase<TH1F>
{
 public:
    
  Comp2RefKolmogorovROOT(void) : Comp2RefBase<TH1F>(){}
  virtual ~Comp2RefKolmogorovROOT(void){}
  /// run the test (result: [0, 1] or <0 for failure)
  float runTest(const TH1F * const h);
  ///get  algorithm name
  static std::string getAlgoName(void){return "Comp2RefKolmogorov";}
  /// true if test cannot run
  bool isInvalid(const TH1F * const h);
 
 protected:
  /// # of bins for test & reference histograms
  Int_t ncx1;
  Int_t ncx2;
  static const Double_t difprec;
};

#endif
