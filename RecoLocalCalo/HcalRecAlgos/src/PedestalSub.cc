#include <iostream>
#include <cmath>
#include <climits>
#include "RecoLocalCalo/HcalRecAlgos/interface/PedestalSub.h"

using namespace std;

PedestalSub::PedestalSub() {
}

PedestalSub::~PedestalSub() { 
}

void PedestalSub::Init(Method method=AvgWithThresh, int runCond=0, float threshold=0.0, float quantile=0.0) {
  fMethod=method;
  fThreshold=threshold;
  fQuantile=quantile;
  fCondition=runCond;
  
  if ( (fMethod==DoNothing||fMethod==AvgWithoutThresh||fMethod==Percentile)&&(fThreshold!=0.0) ) {
    cout << "You are almost certainly doing something wrong. Check your PedestalSub::Calculate parameters, you're passing a threshold for a fMethod that doesn't use them!" << endl;
  }
  else if ( (fMethod!=Percentile)&&(fQuantile!=0.0) ) {
    cout << "You are almost certainly doing something wrong. Check your PedestalSub::Calculate parameters, you're passing a quantile for a fMethod that doesn't use them!" << endl;
  }
  else if ( (fMethod==AvgWithThresh||fMethod==AvgWithThreshNoPedSub)&&(fThreshold==0) ) {
    cout << "You are almost certainly doing something wrong. Check your PedestalSub::Calculate parameters, you're using 0.0 as your threshold!" << endl;
  }
  else if (fMethod==Percentile) {
    //need to fix these parameters;
    if (fCondition==0) {
      fNoisePara=0.92;
    }
    else if (fCondition==50) {
      fNoisePara=2.02;
    }
    else if (fCondition==25) {
      fNoisePara=2.85;
    }
    fNoiseCorr=-inverseGaussCDF(fQuantile);
  }
  
}

void PedestalSub::Calculate(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal, std::vector<double> & corrCharge) const {

  double bseCorr=PedestalSub::GetCorrection(inputCharge, inputPedestal);
  for (Int_t i=0; i<10; i++) {
    if (fMethod==AvgWithThresh||fMethod==Percentile||fMethod==AvgWithoutThresh) {
      corrCharge.push_back(inputCharge[i]-inputPedestal[i]-bseCorr);
    }
    else {
      corrCharge.push_back(inputCharge[i]-bseCorr);
    }
  }
}

double PedestalSub::GetCorrection(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal) const {

  double baseline=0;

  if (fMethod==DoNothing) { 
    baseline=0;
  }
  else if (fMethod==AvgWithThresh) {
    for (Int_t i=0; i<10; i++) {
      if (i==4||i==5) continue;
      if ( (inputCharge[i]-inputPedestal[i])<fThreshold) {
	baseline+=(inputCharge[i]-inputPedestal[i]);
      }
      else {
	baseline+=fThreshold;
      }
    }
    baseline/=8;
  }
  else if (fMethod==AvgWithoutThresh) {
    for (Int_t i=0; i<10; i++) {
      if (i==4||i==5) continue;
      baseline+=(inputCharge[i]-inputPedestal[i]);
    }
    baseline/=8;
  }
  if (fMethod==AvgWithThreshNoPedSub) {
    for (Int_t i=0; i<10; i++) {
      if (i==4||i==5) continue;
      if ( (inputCharge[i])<fThreshold) {
	baseline+=(inputCharge[i]);
      }
      else baseline+=fThreshold;
    }
    baseline/=8;
  }
  else if (fMethod==Percentile) {
    std::vector<float> tempCharge;
    for (int i=0; i<10; i++) {
      if (i==4||i==5||i==6) continue;
      tempCharge.push_back(inputCharge[i]-inputPedestal[i]);
    }
    baseline=sampleQuantile<7>(&tempCharge[0],fQuantile);
    baseline+=fNoisePara*fNoiseCorr;
  }
  
  return baseline;
  
} 
