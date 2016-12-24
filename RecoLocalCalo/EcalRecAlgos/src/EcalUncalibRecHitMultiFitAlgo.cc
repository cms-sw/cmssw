#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitMultiFitAlgo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"

EcalUncalibRecHitMultiFitAlgo::EcalUncalibRecHitMultiFitAlgo() : 
  _computeErrors(true),
  _doPrefit(false),
  _prefitMaxChiSq(1.0),
  _dynamicPedestals(false),
  _mitigateBadSamples(false) { 
    
  _singlebx.resize(1);
  _singlebx << 0;
  
  _pulsefuncSingle.disableErrorCalculation();
  _pulsefuncSingle.setMaxIters(1);
  _pulsefuncSingle.setMaxIterWarnings(false);
    
}

/// compute rechits
EcalUncalibratedRecHit EcalUncalibRecHitMultiFitAlgo::makeRecHit(const EcalDataFrame& dataFrame, const EcalPedestals::Item * aped, const EcalMGPAGainRatio * aGain, const SampleMatrixGainArray &noisecors, const FullSampleVector &fullpulse, const FullSampleMatrix &fullpulsecov, const BXVector &activeBX) {

  uint32_t flags = 0;
  
  const unsigned int nsample = EcalDataFrame::MAXSAMPLES;
  
  double maxamplitude = -std::numeric_limits<double>::max();
  double maxgainratio = -std::numeric_limits<double>::max();
  unsigned int iSampleMax = 0;
  
  double pedval = 0.;
    
  SampleVector amplitudes;
  SampleGainVector gainsNoise;
  SampleGainVector gainsPedestal;
  SampleGainVector badSamples = SampleGainVector::Zero();
  for(unsigned int iSample = 0; iSample < nsample; iSample++) {
    
    const EcalMGPASample &sample = dataFrame.sample(iSample);
    
    double amplitude = 0.;
    int gainId = sample.gainId();
    
    double pedestal = 0.;
    double gainratio = 1.;
        
    if (gainId==0 || gainId==3) {
      pedestal = aped->mean_x1;
      gainratio = aGain->gain6Over1()*aGain->gain12Over6();
      gainsNoise[iSample] = 2;
      gainsPedestal[iSample] = _dynamicPedestals ? 2 : -1;  //-1 for static pedestal
    }
    else if (gainId==1) {
      pedestal = aped->mean_x12;
      gainratio = 1.;
      gainsNoise[iSample] = 0;
      gainsPedestal[iSample] = _dynamicPedestals ? 0 : -1; //-1 for static pedestal
    }
    else if (gainId==2) {
      pedestal = aped->mean_x6;
      gainratio = aGain->gain12Over6();
      gainsNoise[iSample] = 1;
      gainsPedestal[iSample] = _dynamicPedestals ? 1 : -1; //-1 for static pedestals
    }

    if (_dynamicPedestals) {
      amplitude = (double)(sample.adc())*gainratio;
    }
    else {
      amplitude = ((double)(sample.adc()) - pedestal) * gainratio;
    }
    
    if (gainId == 0) {
      //saturation
      if (_dynamicPedestals) {
        amplitude = 4095.*gainratio;
      }
      else {
        amplitude = (4095. - pedestal) * gainratio;
      }
    }
        
    amplitudes[iSample] = amplitude;
    
    if (gainratio > maxgainratio || gainId==0 || amplitude>maxamplitude) {
    //if (iSample==5) {
      maxamplitude = amplitude;
      maxgainratio = gainratio;
      iSampleMax = iSample;
      pedval = pedestal;
    }
        
  }
  
  bool hasGainSwitch = gainsNoise != SampleGainVector::Zero();
  
  //special handling for saturated samples, sample immediately before saturation or for
  //case where maximum sample has gain switched but previous sample has not
  //which is potentially affected by slew rate limitation (EB only)
  //Amplitude is set to zero, and a floating negative single-sample offset is added to the fit
  //to effectively remove the affected sample
  if (_mitigateBadSamples && hasGainSwitch) {
    bool isbarrel = dataFrame.id().subdetId()==EcalBarrel;
    for(unsigned int iSample = 0; iSample < nsample; iSample++) {
      int gainId = dataFrame.sample(iSample).gainId();
      int gainIdNext = iSample<nsample ? dataFrame.sample(iSample+1).gainId() : gainId;
      
      bool isSaturated = gainId==0;
      bool isNextSaturated = gainIdNext==0;
      bool isSlewLimited = isbarrel && iSample==(iSampleMax-1) && gainsNoise.coeff(iSampleMax)>0 && gainsNoise.coeff(iSample)!=gainsNoise.coeff(iSampleMax);
      
      if (isSaturated || isNextSaturated || isSlewLimited) {
        badSamples[iSample] = 1;
      }
    }
  }
  
  double amplitude, amperr, chisq;
  bool status = false;
  
  //compute noise covariance matrix, which depends on the sample gains
  SampleMatrix noisecov;
  if (hasGainSwitch) {
    std::array<double,3> pedrmss = {{aped->rms_x12, aped->rms_x6, aped->rms_x1}};
    std::array<double,3> gainratios = {{ 1., aGain->gain12Over6(), aGain->gain6Over1()*aGain->gain12Over6()}};
    noisecov = SampleMatrix::Zero();
    for (unsigned int gainidx=0; gainidx<noisecors.size(); ++gainidx) {
      SampleGainVector mask = gainidx*SampleGainVector::Ones();
      SampleVector pedestal = (gainsNoise.array()==mask.array()).cast<SampleVector::value_type>();
      if (pedestal.maxCoeff()>0.) {
        //select out relevant components of each correlation matrix, and assume no correlation between samples with
        //different gain
        noisecov += gainratios[gainidx]*gainratios[gainidx]*pedrmss[gainidx]*pedrmss[gainidx]*pedestal.asDiagonal()*noisecors[gainidx]*pedestal.asDiagonal();
      }
    }
  }
  else {
    noisecov = aped->rms_x12*aped->rms_x12*noisecors[0];
  }
  
  //optimized one-pulse fit for hlt
  bool usePrefit = false;
  if (_doPrefit) {
    status = _pulsefuncSingle.DoFit(amplitudes,noisecov,_singlebx,fullpulse,fullpulsecov,gainsPedestal,badSamples);
    amplitude = status ? _pulsefuncSingle.X()[0] : 0.;
    amperr = status ? _pulsefuncSingle.Errors()[0] : 0.;
    chisq = _pulsefuncSingle.ChiSq();
    
    if (chisq < _prefitMaxChiSq) {
      usePrefit = true;
    }
  }
  
  if (!usePrefit) {
  
    if(!_computeErrors) _pulsefunc.disableErrorCalculation();
    status = _pulsefunc.DoFit(amplitudes,noisecov,activeBX,fullpulse,fullpulsecov,gainsPedestal,badSamples);
    chisq = _pulsefunc.ChiSq();
    
    if (!status) {
      edm::LogWarning("EcalUncalibRecHitMultiFitAlgo::makeRecHit") << "Failed Fit" << std::endl;
    }

    unsigned int ipulseintime = 0;
    for (unsigned int ipulse=0; ipulse<_pulsefunc.BXs().rows(); ++ipulse) {
      if (_pulsefunc.BXs().coeff(ipulse)==0) {
        ipulseintime = ipulse;
        break;
      }
    }
    
    amplitude = status ? _pulsefunc.X()[ipulseintime] : 0.;
    amperr = status ? _pulsefunc.Errors()[ipulseintime] : 0.;
  
  }
  
  double jitter = 0.;
  
  //printf("status = %i\n",int(status));
  //printf("amplitude = %5f +- %5f, chisq = %5f\n",amplitude,amperr,chisq);
  
  EcalUncalibratedRecHit rh( dataFrame.id(), amplitude , pedval, jitter, chisq, flags );
  rh.setAmplitudeError(amperr);
  
  if (!usePrefit) {
    for (unsigned int ipulse=0; ipulse<_pulsefunc.BXs().rows(); ++ipulse) {
      int bx = _pulsefunc.BXs().coeff(ipulse);
      if (bx!=0 && std::abs(bx)<100) {
        rh.setOutOfTimeAmplitude(bx+5, status ? _pulsefunc.X().coeff(ipulse) : 0.);
      }
      else if (bx==(100+gainsPedestal[iSampleMax])) {
        rh.setPedestal(status ? _pulsefunc.X().coeff(ipulse) : 0.);
      }
    }
  }
  
  return rh;
}

