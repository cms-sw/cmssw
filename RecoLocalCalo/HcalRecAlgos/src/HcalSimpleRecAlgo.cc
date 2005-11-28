#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSimpleRecAlgo.h"
#include <FWCore/Utilities/interface/Exception.h>

HcalSimpleRecAlgo::HcalSimpleRecAlgo(int firstSample, int samplesToAdd) : firstSample_(firstSample), samplesToAdd_(samplesToAdd) {
}

///Timeshift correction for HPDs based on the position of the peak ADC measurement.
///  Allows for an accurate determination of the relative phase of the pulse shape from
///  the HPD.  Calculated based on a weighted sum of the -1,0,+1 samples relative to the peak
///  as follows:  wpksamp = (0*sample[0] + 1*sample[1] + 2*sample[2]) / (sample[0] + sample[1] + sample[2])
///  where sample[1] is the maximum ADC sample value.
float timeshift_hbheho(float wpksamp) { return 0; }

///Same as above, but for the HF PMTs.
float timeshift_hf(float wpksamp) { return 0; }


namespace HcalSimpleRecAlgoImpl {
  template<class Digi, class RecHit>
  inline RecHit reco(const Digi& digi, const HcalCoder& coder, const HcalCalibrations& calibs, int ifirst, int n) {
    CaloSamples tool;
    coder.adc2fC(digi,tool);

    double ampl=0; int maxI = -1; double maxA = -1e10; float ta=0;
    for (int i=ifirst; i<tool.size() && i<n+ifirst; i++) {
      int capid=digi[i].capid();
      ta = (tool[i]-calibs.pedestal(capid))*calibs.gain(capid);
      ampl+=ta;
      if(ta>maxA){
	maxA=ta;
	maxI=i;
      }
    }

    ////Cannot calculate time value with max ADC sample at first or last position in window....
    if(maxI==0 || maxI==(tool.size()-1)) {
      throw cms::Exception("InvalidRecoParam") << "HcalSimpleRecAlgo::reconstruct :" 
					       << " Invalid max amplitude position, " 
					       << " max Amplitude: "<< maxI
					       << " first: "<<ifirst
					       << " last: "<<(tool.size()-1)
					       << std::endl;
  }


    int capid=digi[maxI-1].capid();
    float t0 = (tool[maxI-1]-calibs.pedestal(capid))*calibs.gain(capid);
    capid=digi[maxI+1].capid();
    float t2 = (tool[maxI+1]-calibs.pedestal(capid))*calibs.gain(capid);    
    float wpksamp = (maxA + 2.0*t2) / (t0 + maxA + t2);
    float time = (maxI - digi.presamples())*25.0 + timeshift_hbheho(wpksamp);
    
    return RecHit(digi.id(),ampl,time);    
  }
}

HBHERecHit HcalSimpleRecAlgo::reconstruct(const HBHEDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<HBHEDataFrame,HBHERecHit>(digi,coder,calibs,firstSample_,samplesToAdd_);
}
HORecHit HcalSimpleRecAlgo::reconstruct(const HODataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<HODataFrame,HORecHit>(digi,coder,calibs,firstSample_,samplesToAdd_);
}

HFRecHit HcalSimpleRecAlgo::reconstruct(const HFDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  CaloSamples tool;
  coder.adc2fC(digi,tool);
  
  double ampl=0; int maxI = -1; double maxA = -1e10; float ta=0;
  for (int i=firstSample_; i<tool.size() && i<samplesToAdd_+firstSample_; i++) {
    int capid=digi[i].capid();
    ta = (tool[i]-calibs.pedestal(capid))*calibs.gain(capid);
    ampl+=ta;
    if(ta>maxA){
      maxA=ta;
      maxI=i;
    }
  }

  ////Cannot calculate time value with max ADC sample at first or last position in window....
  if(maxI==0 || maxI==(tool.size()-1)) {
    throw cms::Exception("InvalidRecoParam") << "HcalSimpleRecAlgo::reconstruct :" 
					 << " Invalid max amplitude position, " 
					 << " max Amplitude: "<< maxI
					 << " first: "<<firstSample_
					 << " last: "<<(tool.size()-1)
					 << std::endl;
  }

  
  int capid=digi[maxI-1].capid();
  float t0 = (tool[maxI-1]-calibs.pedestal(capid))*calibs.gain(capid);
  capid=digi[maxI+1].capid();
  float t2 = (tool[maxI+1]-calibs.pedestal(capid))*calibs.gain(capid);    
  float wpksamp = (maxA + 2.0*t2) / (t0 + maxA + t2);
  float time = (maxI - digi.presamples())*25.0 + timeshift_hf(wpksamp);
  
  return HFRecHit(digi.id(),ampl,time); 
}
