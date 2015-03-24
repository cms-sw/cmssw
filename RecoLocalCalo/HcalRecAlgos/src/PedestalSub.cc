#include <iostream>
#include <cmath>
#include <climits>
#include "RecoLocalCalo/HcalRecAlgos/interface/PedestalSub.h"

using namespace std;

PedestalSub::PedestalSub() : fMethod(AvgWithThresh),fThreshold(2.7),fQuantile(0.0),fCondition(0){
}

PedestalSub::~PedestalSub() { 
}

void PedestalSub::init(Method method=AvgWithThresh, int runCond=0, float threshold=0.0, float quantile=0.0) {
  fMethod=method;
  fThreshold=threshold;
  fQuantile=quantile;
  fCondition=runCond;
}

void PedestalSub::calculate(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal, std::vector<double> & corrCharge) const {

  double bseCorr=PedestalSub::getCorrection(inputCharge, inputPedestal);
  for (auto i=0; i<10; i++) {
    if (fMethod==AvgWithThresh||fMethod==Percentile||fMethod==AvgWithoutThresh) {
      corrCharge.push_back(inputCharge[i]-inputPedestal[i]-bseCorr);
    }
    else {
      corrCharge.push_back(inputCharge[i]-bseCorr);
    }
  }
}

double PedestalSub::getCorrection(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal) const {

  double baseline=0;

  if (fMethod==DoNothing) { 
    baseline=0;
  }
  else if (fMethod==AvgWithThresh) {
    for (auto i=0; i<10; i++) {
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
    for (auto i=0; i<10; i++) {
      if (i==4||i==5) continue;
      baseline+=(inputCharge[i]-inputPedestal[i]);
    }
    baseline/=8;
  }
  if (fMethod==AvgWithThreshNoPedSub) {
    for (auto i=0; i<10; i++) {
      if (i==4||i==5) continue;
      if ( (inputCharge[i])<fThreshold) {
	baseline+=(inputCharge[i]);
      }
      else baseline+=fThreshold;
    }
    baseline/=8;
  }  
  return baseline;
  
} 
