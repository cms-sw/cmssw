#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitTimingCCAlgo.h"

EcalUncalibRecHitTimingCCAlgo::EcalUncalibRecHitTimingCCAlgo(const float startTime, const float stopTime) : _startTime(startTime), _stopTime(stopTime) {}

double EcalUncalibRecHitTimingCCAlgo::computeTimeCC(const EcalDataFrame& dataFrame, const std::vector<double> &amplitudes, const EcalPedestals::Item * aped, const EcalMGPAGainRatio * aGain, const FullSampleVector &fullpulse, EcalUncalibratedRecHit& uncalibRecHit) {
  const unsigned int nsample = EcalDataFrame::MAXSAMPLES;

  double maxamplitude = -std::numeric_limits<double>::max();

  double pulsenorm = 0.;

  std::vector<double> pedSubSamples(nsample);
  for(unsigned int iSample = 0; iSample < nsample; iSample++) {

    const EcalMGPASample &sample = dataFrame.sample(iSample);

    double amplitude = 0.;
    int gainId = sample.gainId();

    double pedestal = 0.;
    double gainratio = 1.;

    if (gainId==0 || gainId==3) {
      pedestal = aped->mean_x1;
      gainratio = aGain->gain6Over1()*aGain->gain12Over6();
    }
    else if (gainId==1) {
      pedestal = aped->mean_x12;
      gainratio = 1.;
    }
    else if (gainId==2) {
      pedestal = aped->mean_x6;
      gainratio = aGain->gain12Over6();
    }

    amplitude = ((double)(sample.adc()) - pedestal) * gainratio;

    if (gainId == 0) {
      //saturation
      amplitude = (4095. - pedestal) * gainratio;
    }

    pedSubSamples.at(iSample) = amplitude;

    if (amplitude>maxamplitude) {
      maxamplitude = amplitude;
    }
    pulsenorm += fullpulse(iSample);
  }

  std::vector<double>::const_iterator amplit;
  for(amplit=amplitudes.begin(); amplit<amplitudes.end(); ++amplit) {
    int ipulse = std::distance(amplitudes.begin(),amplit);
    int bx = ipulse - 5;
    int firstsamplet = std::max(0,bx + 3);
    int offset = 7-3-bx;

    TVectorD pulse;
    pulse.ResizeTo(nsample);
    for (unsigned int isample = firstsamplet; isample<nsample; ++isample) {
      pulse(isample) = fullpulse(isample+offset);
      pedSubSamples.at(isample) = std::max(0., pedSubSamples.at(isample) - amplitudes[ipulse]*pulse(isample)/pulsenorm);
    }
  }

  float globalTimeShift = 100;
  float tStart = _startTime+globalTimeShift;
  float tStop = _stopTime+globalTimeShift;
  float tM = (tStart+tStop)/2;

  float distStart, distStop;
  int counter=0;


  do {
    ++counter;
    distStart = computeCC(pedSubSamples, fullpulse, tStart);
    distStop = computeCC(pedSubSamples, fullpulse, tStop);

    if (distStart > distStop) {
      tStart = tStart;
      tStop = tM;
    }
    else {
      tStart = tM;
      tStop = tStop;
    }
    tM = (tStart+tStop)/2;

    } while ( tStop - tStart > 0.001 && counter<40 );
    // } while ( std::abs((distStart - distStop)/distStop) > 0.0001 && counter<100 );

  tM -= globalTimeShift;

  if (counter<2 || counter>38) {  
    if (counter>15)
      std::cout<<"Counter KUTMF "<<counter<<std::endl;
    tM = 100*25;
  }

  return tM/25;
}

FullSampleVector EcalUncalibRecHitTimingCCAlgo::interpolatePulse(const FullSampleVector& fullpulse, const float t){
  int shift = t/25;
  if (t<0)
    shift -= 1;
  float timeShift = t-25*shift; 
  float tt = timeShift/25;

  // t is in ns
  FullSampleVector interpPulse;
  // Linear
  // for (int i=0; i<fullpulse.size()-1; ++i)
  //       interpPulse[i] = fullpulse[i] + tt*(fullpulse[i+1]-fullpulse[i]);
  // interpPulse[fullpulse.size()-1] = fullpulse[fullpulse.size()-1];

  // 2nd poly
  // 
  // for (int i=1; i<fullpulse.size()-1; ++i)
  //       interpPulse[i] = 0.5*tt*(tt-1)*fullpulse[i-1] - (tt+1)*(tt-1)*fullpulse[i] + 0.5*tt*(tt+1)*fullpulse[i+1];
  // interpPulse[0] = (tt+1)*(tt-1)*fullpulse[0] + 0.5*tt*(tt+1)*fullpulse[1];
  // interpPulse[fullpulse.size()-1] = 0.5*tt*(tt-1)*fullpulse[fullpulse.size()-2] - (tt+1)*(tt-1)*fullpulse[fullpulse.size()-1];

  // 2nd poly with avg
  for (int i=1; i<fullpulse.size()-2; ++i) {
        float a = 0.25*tt*(tt-1)*fullpulse[i-1] + (0.25*(tt-2)-0.5*(tt+1))*(tt-1)*fullpulse[i] + (0.25*(tt+1)-0.5*(tt-2))*tt*fullpulse[i+1] + 0.25*(tt-1)*tt*fullpulse[i+2];
        if (a>0) 
          interpPulse[i] = a;
        else
          interpPulse[i] = 0;
  }
  interpPulse[0] = (0.25*(tt-2) - 0.5*(tt+1))*((tt-1)*fullpulse[0]) + (0.25*(tt+1)+0.5*(tt-2))*tt*fullpulse[1] + 0.25*tt*(tt-1)*fullpulse[2];
  interpPulse[fullpulse.size()-2] = 0.25*tt*(tt-1)*fullpulse[fullpulse.size()-3] + (0.25*(tt-2)-0.5*(tt+1))*(tt-1)*fullpulse[fullpulse.size()-2] + (0.25*(tt+1)-0.5*(tt-2))*tt*fullpulse[fullpulse.size()-1];
  interpPulse[fullpulse.size()-1] = 0.5*tt*(tt-1)*fullpulse[fullpulse.size()-2] - (tt+1)*(tt-1)*fullpulse[fullpulse.size()-1] + (0.25*(tt+1)-0.5*(tt-2))*tt*fullpulse[fullpulse.size()-1];

  FullSampleVector interpPulseShifted;
  for (int i=0; i<interpPulseShifted.size(); ++i) {
      if (i+shift>=0 && i+shift<interpPulse.size())
        interpPulseShifted[i] = interpPulse[i+shift];
      else
        interpPulseShifted[i] = 0;
  }
  return interpPulseShifted;
}

float EcalUncalibRecHitTimingCCAlgo::computeCC(const std::vector<double>& samples, const FullSampleVector& sigmalTemplate, const float& t) {
  int exclude = 1;
  double powerSamples = .0;
  for (int i=exclude; i<int(samples.size()-exclude); ++i)
    powerSamples += std::pow(samples[i],2);

  auto interpolated = interpolatePulse(sigmalTemplate, t);
  double powerTemplate = .0;
  for (int i=exclude; i<int(interpolated.size()-exclude); ++i)
    powerTemplate += std::pow(interpolated[i],2);

  double denominator = std::sqrt(powerTemplate*powerSamples);

  double cc = .0;
  for (int i=exclude; i<int(samples.size()-exclude); ++i){
      cc += interpolated[i]*samples[i];
  }
  return cc/denominator;
}


