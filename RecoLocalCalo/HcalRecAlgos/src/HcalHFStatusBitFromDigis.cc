#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromDigis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm> // for "max"
#include <math.h>
#include <iostream>

HcalHFStatusBitFromDigis::HcalHFStatusBitFromDigis()
{
  // use simple values in default constructor
  minthreshold_=40; // minimum energy threshold (in GeV)

  firstSample_=1; // these are the firstSample, samplesToAdd value of Igor's algorithm -- not necessarily the same as the reco first, toadd values (which are supplied individually for each hit)
  samplesToAdd_=3;
  expectedPeak_=2;

  // Based on Igor V's algorithm:
  //TS4/(TS3+TS4+TS5+TS6) > 0.93 - exp(-0.38275-0.012667*E)
  coef_.push_back(0.93);
  coef_.push_back(-0.38275);
  coef_.push_back(-0.012667);
 // Minimum energy of 10 GeV required
  HFlongwindowEthresh_=40;
  HFlongwindowMinTime_.clear();
  HFlongwindowMinTime_.push_back(-10);
  HFlongwindowMaxTime_.clear();
  HFlongwindowMaxTime_.push_back(8);
  HFshortwindowEthresh_=40;
  HFshortwindowMinTime_.clear();
  HFshortwindowMinTime_.push_back(-10);
  HFshortwindowMaxTime_.clear();
  HFshortwindowMaxTime_.push_back(8);
}

HcalHFStatusBitFromDigis::HcalHFStatusBitFromDigis(const edm::ParameterSet& HFDigiTimeParams,
						   const edm::ParameterSet& HFTimeInWindowParams)
{
  // Specify parameters used in forming the HFDigiTime flag
  firstSample_        = HFDigiTimeParams.getParameter<int>("HFdigiflagFirstSample");
  samplesToAdd_       = HFDigiTimeParams.getParameter<int>("HFdigiflagSamplesToAdd");
  expectedPeak_       = HFDigiTimeParams.getParameter<int>("HFdigiflagExpectedPeak");
  minthreshold_       = HFDigiTimeParams.getParameter<double>("HFdigiflagMinEthreshold");
  coef_              = HFDigiTimeParams.getParameter<std::vector<double> >("HFdigiflagCoef");

  // Specify parameters used in forming HFInTimeWindow flag
  HFlongwindowMinTime_    = HFTimeInWindowParams.getParameter<std::vector<double> >("hflongMinWindowTime");
  HFlongwindowMaxTime_    = HFTimeInWindowParams.getParameter<std::vector<double> >("hflongMaxWindowTime");
  HFlongwindowEthresh_    = HFTimeInWindowParams.getParameter<double>("hflongEthresh");
  HFshortwindowMinTime_    = HFTimeInWindowParams.getParameter<std::vector<double> >("hfshortMinWindowTime");
  HFshortwindowMaxTime_    = HFTimeInWindowParams.getParameter<std::vector<double> >("hfshortMaxWindowTime");
  HFshortwindowEthresh_    = HFTimeInWindowParams.getParameter<double>("hfshortEthresh");
}

HcalHFStatusBitFromDigis::~HcalHFStatusBitFromDigis(){}

void HcalHFStatusBitFromDigis::resetParamsFromDB(int firstSample, int samplesToAdd, int expectedPeak, double minthreshold, const std::vector<double>& coef)
{
  // Resets values based on values in database.
  firstSample_  = firstSample;
  samplesToAdd_ = samplesToAdd;
  expectedPeak_ = expectedPeak;
  minthreshold_ = minthreshold;
  coef_         = coef;
}


void HcalHFStatusBitFromDigis::resetFlagTimeSamples(int firstSample, int samplesToAdd, int expectedPeak)
{
  // This resets the time samples used in the HF flag.  These samples are not necessarily the same 
  // as the flags used by the energy reconstruction
  firstSample_  = firstSample;
  samplesToAdd_ = samplesToAdd;
  expectedPeak_ = expectedPeak;
} // void HcalHFStatusBitFromDigis

void HcalHFStatusBitFromDigis::hfSetFlagFromDigi(HFRecHit& hf, 
						 const HFDataFrame& digi,
						 const HcalCoder& coder,
						 const HcalCalibrations& calib)
{
  // The following 3 values are computed by Igor's algorithm 
  //only in the window [firstSample_, firstSample_ + samplesToAdd_), 
  //which may not be the same as the default reco window.
 
  double totalCharge=0;
  double peakCharge=0;
  double RecomputedEnergy=0;

  CaloSamples tool;
  coder.adc2fC(digi,tool);

  // Compute quantities needed for HFDigiTime, Fraction2TS FLAGS
  for (int i=0;i<digi.size();++i)
    {
      int capid=digi.sample(i).capid();
      double value = tool[i]-calib.pedestal(capid);


      // Sum all charge within flagging window, find charge in expected peak time slice
      if (i >=firstSample_ && i < firstSample_+samplesToAdd_)
	{
	  totalCharge+=value;
	  RecomputedEnergy+=value*calib.respcorrgain(capid);
	  if (i==expectedPeak_) peakCharge=value;
	}
    } // for (int i=0;i<digi.size();++i)

  // FLAG:  HcalCaloLabels::HFDigiTime
  // Igor's algorithm:  compare charge in peak to total charge in window
  if (RecomputedEnergy>=minthreshold_)  // don't set noise flags for cells below a given threshold
    {
      // Calculate allowed minimum value of (TS4/TS3+4+5+6):
      double cutoff=0; // no arguments specified; no cutoff applied
      if (coef_.size()>0)
	cutoff=coef_[0];
      // default cutoff takes the form:
      // cutoff = coef_[0] - exp(coef_[1]+coef_[2]*E+coef_[3]*E^2+...)
      double powRE=1;
      double expo_arg=0;
      for (unsigned int zz=1;zz<coef_.size();++zz)
	{
	  expo_arg+=coef_[zz]*powRE;
	  powRE*=RecomputedEnergy;
	}
      cutoff-=exp(expo_arg);
      
      if (peakCharge/totalCharge<cutoff)
	hf.setFlagField(1,HcalCaloFlagLabels::HFDigiTime);
    }

  // FLAG:  HcalCaloLabels:: HFInTimeWindow
  // Timing algorithm
  if (hf.id().depth()==1)
    {
      if (hf.energy()>=HFlongwindowEthresh_)
	{
	  float mult=1./hf.energy();
	  float enPow=1.;
	  float mintime=0;
	  float maxtime=0;
	  for (unsigned int i=0;i<HFlongwindowMinTime_.size();++i)
	    {
	      mintime+=HFlongwindowMinTime_[i]*enPow;
	      maxtime+=HFlongwindowMaxTime_[i]*enPow;
	      enPow*=mult;
	    }
	  if (hf.time()<mintime || hf.time()>maxtime)
	    hf.setFlagField(1,HcalCaloFlagLabels::HFInTimeWindow);
	}
    }
  else if (hf.id().depth()==2)
    {
      if (hf.energy()>=HFshortwindowEthresh_)
	{
	  float mult=1./hf.energy();
	  float enPow=1.;
	  float mintime=0;
	  float maxtime=0;
	  for (unsigned int i=0;i<HFshortwindowMinTime_.size();++i)
	    {
	      mintime+=HFshortwindowMinTime_[i]*enPow;
	      maxtime+=HFshortwindowMaxTime_[i]*enPow;
	      enPow*=mult;
	    }
	  if (hf.time()<mintime || hf.time()>maxtime)
	    hf.setFlagField(1,HcalCaloFlagLabels::HFInTimeWindow);
	}
    }

  return;
}

