#include <iostream>
#include <cmath>
#include <climits>
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalDeterministicFit.h"

constexpr float HcalDeterministicFit::invGpar[3];
constexpr float HcalDeterministicFit::negThresh[2];
constexpr int HcalDeterministicFit::HcalRegion[2];
constexpr float HcalDeterministicFit::rCorr[2];

using namespace std;

HcalDeterministicFit::HcalDeterministicFit() {
}

HcalDeterministicFit::~HcalDeterministicFit() { 
}

void HcalDeterministicFit::init(HcalTimeSlew::ParaSource tsParam, HcalTimeSlew::BiasSetting bias, NegStrategy nStrat, PedestalSub pedSubFxn_, std::vector<double> pars, double respCorr) {
  for(int fi=0; fi<9; fi++){
	fpars[fi] = pars.at(fi);
  }

  fTimeSlew=tsParam;
  fTimeSlewBias=bias;
  fNegStrat=nStrat;
  fPedestalSubFxn_=pedSubFxn_;
  frespCorr=respCorr;
}

constexpr float HcalDeterministicFit::landauFrac[];
// Landau function integrated in 1 ns intervals
//Landau pulse shape from https://indico.cern.ch/event/345283/contribution/3/material/slides/0.pdf
//Landau turn on by default at left edge of time slice 
// normalized to 1 on [0,10000]
void HcalDeterministicFit::getLandauFrac(float tStart, float tEnd, float &sum) const{

  if (std::abs(tStart-tEnd-tsWidth)<0.1) {
    sum=0;
    return;
  }
  sum= landauFrac[int(ceil(tStart+tsWidth))];
  return;
}
