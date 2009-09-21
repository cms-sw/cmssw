#ifndef CalibTracker_SiStripLorentzAngle_SymmetryFit                                                        
#define CalibTracker_SiStripLorentzAngle_SymmetryFit           

#include "TH1.h"
#include "TF1.h"
#include <string>

class SymmetryFit {

 public:
  
  static TH1* symmetryChi2(const TH1*, const std::pair<unsigned,unsigned>);
  static std::string name(std::string base) {return base+"_symmchi2";}
  static TF1* fitfunction();
  
 private:

  SymmetryFit(const TH1*, const std::pair<unsigned,unsigned>, const unsigned);
  float chi2_element(std::pair<unsigned,unsigned>);
  float chi2(std::pair<unsigned,unsigned>);
  void makeChi2Histogram();
  void fillchi2();
  int fit();

  const TH1* symm_candidate_;
  const std::pair<unsigned,unsigned> range_;
  const unsigned ndf_;
  TH1* chi2_;

};

#endif
