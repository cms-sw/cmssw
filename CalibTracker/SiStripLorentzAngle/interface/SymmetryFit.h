#ifndef CalibTracker_SiStripLorentzAngle_SymmetryFit                                                        
#define CalibTracker_SiStripLorentzAngle_SymmetryFit           

#include "TH1.h"
#include "TF1.h"
#include <string>

class SymmetryFit {

 public:
  
  static TH1* symmetryChi2(std::string, const std::vector<TH1*>&, const std::pair<unsigned,unsigned>);
  static TH1* symmetryChi2(const TH1*, const std::pair<unsigned,unsigned>);
  static std::string name(std::string base) {return base+"_symmchi2";}
  static TF1* fitfunction();
  static std::vector<double> pol2_from_pol2(TH1* hist);
  static std::vector<double> pol2_from_pol3(TH1* hist);
  
 private:

  SymmetryFit(const TH1*, const std::pair<unsigned,unsigned>);
  std::pair<unsigned,unsigned> findUsableMinMax() const;
  std::vector<std::pair<unsigned,unsigned> > continuousRanges() const;
  float chi2_element(std::pair<unsigned,unsigned>);
  float chi2(std::pair<unsigned,unsigned>);
  void makeChi2Histogram();
  void fillchi2();
  int fit();
  SymmetryFit operator+=(const SymmetryFit& R) { ndf_+=R.ndf_; chi2_->Add(R.chi2_); return *this;}

  const TH1* symm_candidate_;
  const unsigned minDF_;
  const std::pair<unsigned,unsigned> range_,minmaxUsable_;
  unsigned ndf_;
  TH1* chi2_;

};

#endif
