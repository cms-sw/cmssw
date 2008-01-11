#ifndef _MEANWITHINEXPECTED_ROOT_H
#define _MEANWITHINEXPECTED_ROOT_H

#include "DQMServices/Core/interface/QualTestBase.h"

/// algorithm for testing if histogram's mean value is near expected value
class MeanWithinExpectedROOT : public SimpleTest<TH1F>
{
 public:
    
  MeanWithinExpectedROOT(void) : SimpleTest<TH1F>()
  {validMethod_ = validExpMean_ = false;}
  virtual ~MeanWithinExpectedROOT(void){}

  /// set expected value for mean
  void setExpectedMean(float exp_mean)
  {exp_mean_ = exp_mean; validExpMean_ = true;}

  void useRMS(void){useRMS_ = true; useSigma_ = useRange_ = false;
    validMethod_ = true;}
  void useSigma(float expected_sigma)
  {useSigma_ = true; useRMS_ = useRange_ = false; 
    sigma_ = expected_sigma; checkSigma();}
  void useRange(float xmin, float xmax)
  {useRange_ = true; useSigma_ = useRMS_ = false;
    xmin_ = xmin; xmax_ = xmax; checkRange();}
  
  /** run the test;
     (a) if useRange is called: 1 if mean within allowed range, 0 otherwise
     
     (b) is useRMS or useSigma is called: result is the probability 
     Prob(chi^2, ndof=1) that the mean of histogram will be deviated by more than
     +/- delta from <expected_mean>, where delta = mean - <expected_mean>, and
     chi^2 = (delta/sigma)^2. sigma is the RMS of the histogram ("useRMS") or
     <expected_sigma> ("useSigma")
     e.g. for delta = 1, Prob = 31.7%
          for delta = 2, Prob = 4.55%

     (returns result in [0, 1] or <0 for failure) */
  float runTest(const TH1F * const h);
  ///get  algorithm name
  static std::string getAlgoName(void){return "MeanWithinExpected";}
  /// true if test cannot run
  bool isInvalid(void);

 protected:
  /// if true, will use RMS of distribution
  bool useRMS_;
  /// if true, will use expected_sigma
  bool useSigma_;
  /// if true, will use allowed range
  bool useRange_;

  /// check that exp_sigma_ is non-zero
  void checkSigma(void);
  /// check that allowed range is logical
  void checkRange(void);

  /// test for useRange_ = true case
  float doRangeTest(const TH1F * const h);
  /// test assuming mean value is quantity with gaussian errors
  float doGaussTest(const TH1F * const h, float sigma);

  /// sigma to be used in probability calculation (use only if useSigma_ = true)
  float sigma_;
  /// expected mean value (used only if useSigma_ = true or useRMS_ = true)
  float exp_mean_;
  /// allowed range for mean (use only if useRange_ = true)
  float xmin_, xmax_;

  /// true if method has been chosen
  bool validMethod_;
  /// true if expected mean has been chosen
  bool validExpMean_;

};

#endif
