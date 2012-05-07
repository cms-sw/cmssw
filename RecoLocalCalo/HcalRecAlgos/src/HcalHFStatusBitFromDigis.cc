#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromDigis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm> // for "max"
#include <math.h>
#include <iostream>

HcalHFStatusBitFromDigis::HcalHFStatusBitFromDigis()
{
  // use simple values in default constructor
  minthreshold_=40; // minimum energy threshold (in GeV)
  recoFirstSample_=0;
  recoSamplesToAdd_=10;
  firstSample_=3;
  samplesToAdd_=4;
  expectedPeak_=4;
  // Based on Igor V's algorithm:
  //TS4/(TS3+TS4+TS5+TS6) > 0.93 - exp(-0.38275-0.012667*E)
  coef0_= 0.93;
  coef1_ = -0.38275;
  coef2_ = -0.012667;
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

HcalHFStatusBitFromDigis::HcalHFStatusBitFromDigis(int recoFirstSample,
						   int recoSamplesToAdd,
						   const edm::ParameterSet& HFDigiTimeParams,
						   const edm::ParameterSet& HFTimeInWindowParams)
{
  recoFirstSample_    = recoFirstSample;
  recoSamplesToAdd_   = recoSamplesToAdd;

  // Specify parameters used in forming the HFDigiTime flag
  firstSample_        = HFDigiTimeParams.getParameter<int>("HFdigiflagFirstSample");
  samplesToAdd_       = HFDigiTimeParams.getParameter<int>("HFdigiflagSamplesToAdd");
  expectedPeak_       = HFDigiTimeParams.getParameter<int>("HFdigiflagExpectedPeak");
  minthreshold_       = HFDigiTimeParams.getParameter<double>("HFdigiflagMinEthreshold");
  coef0_              = HFDigiTimeParams.getParameter<double>("HFdigiflagCoef0");
  coef1_              = HFDigiTimeParams.getParameter<double>("HFdigiflagCoef1");
  coef2_              = HFDigiTimeParams.getParameter<double>("HFdigiflagCoef2");

  // Specify parameters used in forming HFInTimeWindow flag
  HFlongwindowMinTime_    = HFTimeInWindowParams.getParameter<std::vector<double> >("hflongMinWindowTime");
  HFlongwindowMaxTime_    = HFTimeInWindowParams.getParameter<std::vector<double> >("hflongMaxWindowTime");
  HFlongwindowEthresh_    = HFTimeInWindowParams.getParameter<double>("hflongEthresh");
  HFshortwindowMinTime_    = HFTimeInWindowParams.getParameter<std::vector<double> >("hfshortMinWindowTime");
  HFshortwindowMaxTime_    = HFTimeInWindowParams.getParameter<std::vector<double> >("hfshortMaxWindowTime");
  HFshortwindowEthresh_    = HFTimeInWindowParams.getParameter<double>("hfshortEthresh");
}

HcalHFStatusBitFromDigis::~HcalHFStatusBitFromDigis(){}

void HcalHFStatusBitFromDigis::hfSetFlagFromDigi(HFRecHit& hf, 
						 const HFDataFrame& digi,
						 const HcalCoder& coder,
						 const HcalCalibrations& calib)
{
  // The following 3 values are computed using the default reconstruction window (for Shuichi's algorithm)
  double maxInWindow=-10; // maximum value found in reco window
  int maxCapid=-1;
  int maxTS=-1;  // time slice where maximum is found

  // The following 3 values are computed only in the window (firstSample_, firstSample_ + samplesToAdD_), which may not be the same as the default reco window  (for Igor's algorithm)
  double totalCharge=0;
  double peakCharge=0;
  double RecomputedEnergy=0;

  CaloSamples tool;
  coder.adc2fC(digi,tool);

  // Compute quantities needed for HFDigiTime, Fraction2TS FLAGS
  for (int i=0;i<digi.size();++i)
    {
      int capid=digi.sample(i).capid();
      //double value = digi.sample(i).nominal_fC()-calib.pedestal(capid);
      double value = tool[i]-calib.pedestal(capid);
      // Find largest value within reconstruction window
      if (i>=recoFirstSample_ && i <recoFirstSample_+recoSamplesToAdd_)
	{
	  // Find largest overall pulse within the full digi, or just the allowed window?
	  if (value>maxInWindow) 
	    {
	      maxCapid=capid;
	      maxInWindow=value;  
	      maxTS=i;
	    }
	}

      // Sum all charge within flagging window, find charge in expected peak time slice
      if (i >=firstSample_ && i < firstSample_+samplesToAdd_)
	{
	  totalCharge+=value;
	  RecomputedEnergy+=value*calib.respcorrgain(capid);
	  if (i==expectedPeak_) peakCharge=value;
	}
    } // for (int i=0;i<digi.size();++i)
  
  // FLAG:  HcalCaloLabel::Fraction2TS
  // Shuichi's Algorithm:  Compare size of peak in digi to charge in TS immediately before peak
  int TSfrac_counter=1; 
  // get pedestals for each capid -- add 4 to each capid, and then check mod 4.
  // (This takes care of the case where max capid =0 , and capid-1 would then be negative)
  if (maxTS>0 &&
      tool[maxTS]!=calib.pedestal(maxCapid))
    TSfrac_counter=int(50*((tool[maxTS-1]-calib.pedestal((maxCapid+3)%4))/(tool[maxTS]-calib.pedestal((maxCapid+4)%4)))+1); // 6-bit counter to hold peak ratio info
  hf.setFlagField(TSfrac_counter, HcalCaloFlagLabels::Fraction2TS,6);

  // FLAG:  HcalCaloLabels::HFDigiTime
  // Igor's algorithm:  compare charge in peak to total charge in window
  if (RecomputedEnergy>=minthreshold_)  // don't set noise flags for cells below a given threshold
    {
      // Calculate allowed minimum value of (TS4/TS3+4+5+6):
      double cutoff=coef0_-exp(coef1_+coef2_*RecomputedEnergy);
      
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

