#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromDigis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm> // for "max"
#include <math.h>
#include <iostream>
using namespace std;

HcalHFStatusBitFromDigis::HcalHFStatusBitFromDigis()
{
  // use simple values in default constructor
  HFpulsetimemin_     = 0;
  HFpulsetimemax_     = 10;
  HFratio_beforepeak_ = .1;
  HFratio_afterpeak_  = 1.;
  adcthreshold_=10; // minimum (pedestal-subtracted) ADC value needed for a cell to be considered noisy
}

HcalHFStatusBitFromDigis::HcalHFStatusBitFromDigis(int HFpulsetimemin,int HFpulsetimemax, double HFratiobefore, double HFratioafter, int adcthreshold)
{
  HFpulsetimemin_     = HFpulsetimemin;
  HFpulsetimemax_     = HFpulsetimemax;
  HFratio_beforepeak_ = HFratiobefore;
  HFratio_afterpeak_  = HFratioafter;
  adcthreshold_       = adcthreshold; 
}

HcalHFStatusBitFromDigis::~HcalHFStatusBitFromDigis(){}

void HcalHFStatusBitFromDigis::hfSetFlagFromDigi(HFRecHit& hf, const HFDataFrame& digi)
{
  int status=0;
  int maxtime=0;
  int maxval=-3;  // maxval is 'pedestal subtracted', with default pedestal of 3 ADC counts

  for (int i=0;i<digi.size();++i)
    {
      if (digi.sample(i).adc()>maxval) // need to make pedestal subtraction at some point
	{
	  maxtime=i;
	  maxval=digi.sample(i).adc()-3;  // assume all pedestal means are 3 ADC counts;  some day we can be more clever about pedestal subtraction
	}
    }
  
  if (maxval<adcthreshold_) return; // don't set noise flags for cells below a given threshold

  // Check that peak occurs in correct time window
  if (maxtime<HFpulsetimemin_ || maxtime>HFpulsetimemax_)
    status=1;
    
  // Check that peak is >= time slice prior to peak
  else if (maxtime>0 && (float)(digi.sample(maxtime-1).adc()-3)/maxval>=HFratio_beforepeak_)
    status=1;

  // Check that peak is >= time slice after peak
  /* 
     require >= so that a hot cell (where all digi sample values are the same)
     will still get marked as noisy if HFratio_afterpeak_ ==1 
  */
  else if (maxtime<(digi.size()-1) && (float)(digi.sample(maxtime+1).adc()-3)/maxval>=HFratio_afterpeak_)
    status=1;

  // set flag starting at index HFDigiTime, with index of 1
  hf.setFlagField(status,HcalCaloFlagLabels::HFDigiTime, 1);

  return;
}

