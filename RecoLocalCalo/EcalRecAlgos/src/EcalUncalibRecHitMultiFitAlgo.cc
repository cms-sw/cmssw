#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitMultiFitAlgo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"

EcalUncalibRecHitMultiFitAlgo::EcalUncalibRecHitMultiFitAlgo() : 
  _computeErrors(true) { 
}

/// compute rechits
EcalUncalibratedRecHit EcalUncalibRecHitMultiFitAlgo::makeRecHit(const EcalDataFrame& dataFrame, const EcalPedestals::Item * aped, const EcalMGPAGainRatio * aGain, const TMatrixDSym &noisecor, const TVectorD &fullpulse, const TMatrixDSym &fullpulsecov, const std::set<int> &activeBX) {

  uint32_t flags = 0;
  
  const unsigned int nsample = EcalDataFrame::MAXSAMPLES;
  
  double maxamplitude = -std::numeric_limits<double>::max();
  
  double pedval = 0.;
  double pedrms = 0.;
  
  std::vector<double> amplitudes(nsample);
  for(unsigned int iSample = 0; iSample < nsample; iSample++) {
    
    const EcalMGPASample &sample = dataFrame.sample(iSample);
    
    double amplitude = 0.;
    int gainId = sample.gainId();
    
    double pedestal = 0.;
    double pederr = 0.;
    double gainratio = 1.;
        
    if (gainId==0 || gainId==3) {
      pedestal = aped->mean_x1;
      pederr = aped->rms_x1;
      gainratio = aGain->gain6Over1()*aGain->gain12Over6();
    }
    else if (gainId==1) {
      pedestal = aped->mean_x12;
      pederr = aped->rms_x12;
      gainratio = 1.;
    }
    else if (gainId==2) {
      pedestal = aped->mean_x6;
      pederr = aped->rms_x6;
      gainratio = aGain->gain12Over6();
    }

    amplitude = ((double)(sample.adc()) - pedestal) * gainratio;
    
    if (gainId == 0) {
      //saturation
      amplitude = (4095. - pedestal) * gainratio;
    }
        
    amplitudes[iSample] = amplitude;
    
    if (amplitude>maxamplitude) {
    //if (iSample==5) {
      maxamplitude = amplitude;
      pedval = pedestal;
      pedrms = pederr*gainratio;
    }    
        
  }
    
  
  std::vector<double> fitvals;
  std::vector<double> fiterrs;
  
  if(!_computeErrors) _pulsefunc.disableErrorCalculation();
  bool status = _pulsefunc.DoFit(amplitudes,noisecor,pedrms,activeBX,fullpulse,fullpulsecov);
  double chisq = _pulsefunc.ChiSq();
  
  if (!status) {
    edm::LogWarning("EcalUncalibRecHitMultiFitAlgo::makeRecHit") << "Failed Fit" << std::endl;
  }

  unsigned int ipulseintime = std::distance(activeBX.begin(),activeBX.find(0));
  double amplitude = status ? _pulsefunc.X()[ipulseintime] : 0.;
  double amperr = status ? _pulsefunc.Errors()[ipulseintime] : 0.;
  
  double jitter = 0.;
  
  //printf("amplitude = %5f +- %5f, chisq = %5f\n",amplitude,amperr,chisq);
  
  EcalUncalibratedRecHit rh( dataFrame.id(), amplitude , pedval, jitter, chisq, flags );
  rh.setAmplitudeError(amperr);
  for (std::set<int>::const_iterator bxit = activeBX.begin(); bxit!=activeBX.end(); ++bxit) {
    int ipulse = std::distance(activeBX.begin(),bxit);
    if(*bxit==0) {
      rh.setOutOfTimeAmplitude(*bxit+5,0.);
    } else {
      rh.setOutOfTimeAmplitude(*bxit+5, status ? _pulsefunc.X()[ipulse] : 0.);
    }
  }

  return rh;
}

