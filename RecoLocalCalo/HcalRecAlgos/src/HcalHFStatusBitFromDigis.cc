#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromDigis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm> // for "max"
#include <math.h>
#include <iostream>
using namespace std;

HcalHFStatusBitFromDigis::HcalHFStatusBitFromDigis()
{
  // use simple values in default constructor
  minthreshold_=10; // minimum total fC (summed over allowed range of time slices) needed for an HF channel to be considered noisy
  firstSample_=3;
  samplesToAdd_=4;
  expectedPeak_=4;
  // Based on Igor V's algorithm:
  //TS4/(TS3+TS4+TS5+TS6) > 0.93 - exp(-0.38275-0.012667*E)
  coef0_= 0.93;
  coef1_ = -0.38275;
  coef2_ = -0.012667;
}

HcalHFStatusBitFromDigis::HcalHFStatusBitFromDigis(int firstSample, 
						   int samplesToAdd, 
						   int expectedPeak,
						   double minthreshold,
						   double coef0, double coef1, double coef2)
{
  firstSample_        = firstSample;
  samplesToAdd_       = samplesToAdd;
  expectedPeak_       = expectedPeak;
  minthreshold_       = minthreshold;
  coef0_              = coef0;
  coef1_              = coef1;
  coef2_              = coef2;
}

HcalHFStatusBitFromDigis::~HcalHFStatusBitFromDigis(){}

void HcalHFStatusBitFromDigis::hfSetFlagFromDigi(HFRecHit& hf, 
						 const HFDataFrame& digi,
						 const HcalCalibrations& calib)
{
  int status=0;
  int maxtime=0;
  int maxval=-10;  // maxval is 'pedestal subtracted', with default pedestal of 3 ADC counts

  int maxInWindow=0; // maximum value found in reco window
  int maxCapid=-1;

  double totalCharge=0;
  double peakCharge=0;
  for (int i=0;i<digi.size();++i)
    {
      int capid=digi.sample(i).capid();
      double value = digi.sample(i).nominal_fC()-calib.pedestal(capid);

      // Check for largest pulse within reco window
      if (i >=firstSample_ && i < samplesToAdd_ && i < digi.size())
	{
	  totalCharge+=value;
	  if (i==expectedPeak_) peakCharge=value;
	}
      
      // Find largest overall pulse within the full digi, not just the allowed window?
      if (value>maxval) 
	{
	  maxtime=i;
	  maxCapid=capid;
	  maxval=value;  // get pedestal directly for each capid
	}
    }
  
  // Compare size of peak in reco window to charge in TS immediately before peak
  int TSfrac_counter=1; 
  // get pedestals for each capid -- add 4 to each capid, and then check mod 4.
  // (This takes care of the case where max capid =0 , and capid-1 would then be negative)
  if (maxInWindow>0 && digi[maxInWindow].nominal_fC()!=3)
    TSfrac_counter=int(50*((digi[maxInWindow-1].nominal_fC()-calib.pedestal((maxCapid+3)%4))/(digi[maxInWindow].nominal_fC()-calib.pedestal((maxCapid+4)%4)))+1); // 6-bit counter to hold peak ratio info
  hf.setFlagField(TSfrac_counter, HcalCaloFlagLabels::Fraction2TS,6);

  if (maxval<minthreshold_) return; // don't set noise flags for cells below a given threshold?

  // Calculate allowed minimum value of (TS4/TS3+4+5+6):
  double cutoff=coef0_-exp(coef1_-coef2_*hf.energy());
  
  if (peakCharge/totalCharge<cutoff)
    status=1;
  else
    status=0;

  // set flag  at index HFDigiTime
  hf.setFlagField(status,HcalCaloFlagLabels::HFDigiTime);

  return;
}

