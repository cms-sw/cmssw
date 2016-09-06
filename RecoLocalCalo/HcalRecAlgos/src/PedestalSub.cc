#include <iostream>
#include <cmath>
#include <climits>
#include "RecoLocalCalo/HcalRecAlgos/interface/PedestalSub.h"

using namespace std;

PedestalSub::PedestalSub() : fThreshold(2.7),fQuantile(0.0),fCondition(0){
}

PedestalSub::~PedestalSub() { 
}

void PedestalSub::init(int runCond=0, float threshold=0.0, float quantile=0.0) {
  fThreshold=threshold;
  fQuantile=quantile;
  fCondition=runCond;
}

void PedestalSub::calculate(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal, std::vector<double> & corrCharge) const {

  double bseCorr=PedestalSub::getCorrection(inputCharge, inputPedestal);
  for (auto i=0; i<10; i++) {
      corrCharge.push_back(inputCharge[i]-inputPedestal[i]-bseCorr);
  }
}

double PedestalSub::getCorrection(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal) const {

  double baseline=0;

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
  return baseline;
  
} 
