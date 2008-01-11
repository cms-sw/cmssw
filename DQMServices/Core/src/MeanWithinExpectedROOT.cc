#include "DQMServices/Core/interface/MeanWithinExpectedROOT.h"

#include <TMath.h>

#include <iostream>

using std::cerr; using std::endl;

/* run the test;
   (a) if useRange is called: 1 if mean within allowed range, 0 otherwise
   
   (b) is useRMS or useSigma is called: result is the probability 
   Prob(chi^2, ndof=1) that the mean of histogram will be deviated by more than
   +/- delta from <expected_mean>, where delta = mean - <expected_mean>, and
   chi^2 = (delta/sigma)^2. sigma is the RMS of the histogram ("useRMS") or
   <expected_sigma> ("useSigma")
   e.g. for delta = 1, Prob = 31.7%
   for delta = 2, Prob = 4.55%
   
   (returns result in [0, 1] or <0 for failure) */
float MeanWithinExpectedROOT::runTest(const TH1F * const h)
{
  if(!h) return -1;
  if(isInvalid())return -1;
  
  if(useRange_)
    return doRangeTest(h);

  if(useSigma_)
    return doGaussTest(h, sigma_);
  
  if(useRMS_)
    return doGaussTest(h, h->GetRMS());

  // we should never reach this point;
  return -99;
}

// test assuming mean value is quantity with gaussian errors
float MeanWithinExpectedROOT::doGaussTest(const TH1F * const h, float sigma)
{
  float chi = (h->GetMean() - exp_mean_)/sigma;
  return TMath::Prob(chi*chi, 1);
}

// test for useRange_ = true case
float MeanWithinExpectedROOT::doRangeTest(const TH1F * const h)
{
  float mean = h->GetMean();
  if(mean <= xmax_ && mean >= xmin_)
    return 1;
  else
    return 0;  
}

// check that exp_sigma_ is non-zero
void MeanWithinExpectedROOT::checkSigma(void)
{
  if(sigma_ != 0)
    validMethod_ = true;
  else
    {
      cerr << " *** Error! Expected sigma = " << sigma_ << " in algorithm "
	   << getAlgoName() << endl;
    validMethod_ = false;
    }
}

// check that allowed range is logical
void MeanWithinExpectedROOT::checkRange(void)
{
  if(xmin_ < xmax_)
    validMethod_ = true;
  else
    {
      cerr << " *** Error! Illogical range: (" << xmin_ << ", " << xmax_ 
	   << ") in algorithm " << getAlgoName() << endl;
      validMethod_ = false;
    }
}

// true if test cannot run
bool MeanWithinExpectedROOT::isInvalid(void)
{
  // if useRange_ = true, test does not need a "expected mean value"
  if(useRange_)
    return !validMethod_; // set by checkRange()

  // otherwise (useSigma_ or useRMS_ case), we also need to check 
  // if "expected mean value" has been set
  return !validMethod_  // set by useRMS() or checkSigma()
    || !validExpMean_; // set by setExpectedMean()

}
