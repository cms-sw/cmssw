#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitMultiFitAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/PulseChiSqSNNLS.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"

/// compute rechits
EcalUncalibratedRecHit EcalUncalibRecHitMultiFitAlgo::makeRecHit(const EcalDataFrame& dataFrame, const EcalPedestals::Item * aped, const EcalMGPAGainRatio * aGain, const TMatrixDSym &noisecor, const TVectorD &fullpulse, const TMatrixDSym &fullpulsecov, std::set<int> activeBX) {

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
  double chisq = 0.;

      
  TMatrixDSym noisecov = pedrms*pedrms*noisecor;
  
  PulseChiSqSNNLS pulsefuncnnls(amplitudes,noisecov,activeBX,fullpulse,fullpulsecov);
  const int maxiter = 50;
  int nnlsiter = 0;
  double chisqnnls = 0.;
  while (true) {
    pulsefuncnnls.Minimize();
    if (nnlsiter>=maxiter) break;
    
    double chisqnow = pulsefuncnnls.ChiSq();
    double deltachisq = chisqnow-chisqnnls;
    chisqnnls = chisqnow;
    if (std::abs(deltachisq)<1e-3) {
      break;
    }
    ++nnlsiter;
    pulsefuncnnls.updateCov(pulsefuncnnls.X(),noisecov,activeBX,fullpulsecov);    
  }

  unsigned int ipulseintime = std::distance(activeBX.begin(),activeBX.find(0));
  double amplitude = pulsefuncnnls.X()[ipulseintime];
  //double amperr = status ? minim.Errors()[ipulseintime] : 0.;
  double amperr = 0.;  
  
  double jitter = 0.;
  
  EcalUncalibratedRecHit rh( dataFrame.id(), amplitude , pedval, jitter, chisq, flags );
  rh.setAmplitudeError(amperr);
  for(unsigned int ipulse=0; ipulse<activeBX.size(); ++ipulse) {
    if(ipulse==ipulseintime) {
      rh.setOutOfTimeAmplitude(ipulse,0.);
      rh.setOutOfTimeAmplitudeError(ipulse,0.);
    } else {
      rh.setOutOfTimeAmplitude(ipulse, pulsefuncnnls.X()[ipulse]);
      //rh.setOutOfTimeAmplitudeError(ipulse, status ? minim.Errors()[ipulse] : 0.);
      rh.setOutOfTimeAmplitudeError(ipulse, 0.);
    }
  }

  return rh;
}

