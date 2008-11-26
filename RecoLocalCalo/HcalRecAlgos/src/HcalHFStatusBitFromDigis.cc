#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromDigis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm> // for "max"
#include <math.h>

HcalHFStatusBitFromDigis::HcalHFStatusBitFromDigis()
{
  // use simple values in default constructor
  HFpulsetimemin_     = 0;
  HFpulsetimemax_     = 10;
  HFratio_beforepeak_ = .1;
  HFratio_afterpeak_  = 1.;
  bit_=0;
}

HcalHFStatusBitFromDigis::HcalHFStatusBitFromDigis(int HFpulsetimemin,int HFpulsetimemax, double HFratiobefore, double HFratioafter, int bit)
{
  HFpulsetimemin_     = HFpulsetimemin;
  HFpulsetimemax_     = HFpulsetimemax;
  HFratio_beforepeak_ = HFratiobefore;
  HFratio_afterpeak_  = HFratioafter;
  bit_=bit;
}

HcalHFStatusBitFromDigis::~HcalHFStatusBitFromDigis(){}

void HcalHFStatusBitFromDigis::hfSetFlagFromDigi(HFRecHit& hf, const HFDataFrame& digi)
{
  int status=0;
  int maxtime=0;
  int maxval=-3;

  for (int i=0;i<digi.size();++i)
    {
      if (digi.sample(i).adc()>maxval) // need to make pedestal subtraction at some point
	{
	  maxtime=i;
	  maxval=digi.sample(i).adc()-3;  // assume all pedestal means are 3 ADC counts;  some day we can be more clever about pedestal subtraction
	}
    }
  // Check that peak occurs in correct time window
  if (maxtime<HFpulsetimemin_ || maxtime>HFpulsetimemax_)
    status=1;
    
  // Check that peak is >> time slice prior to peak
  else if (maxtime>0 && (float)digi.sample(maxtime-1).adc()/maxval>HFratio_beforepeak_)
    status=1;

  // Check that peak is >> time slice after peak
  else if (maxtime<digi.size() && (float)digi.sample(maxtime+1).adc()/maxval>HFratio_afterpeak_)
    status=1;

  // Set flag by forming an 'or' of status with old flag value
  hf.setFlags(hf.flags()|(status<<bit_));
  return;
}

