#include <iostream>
#include <cmath>
#include <climits>
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalDeterministicFit.h"
#include "TMath.h"

using namespace std;

HcalDeterministicFit::HcalDeterministicFit() {
}

HcalDeterministicFit::~HcalDeterministicFit() { 
}

void HcalDeterministicFit::init(HcalTimeSlew::ParaSource tsParam, HcalTimeSlew::BiasSetting bias, NegStrategy nStrat, PedestalSub pedSubFxn_, double parhb0, double parhb1, double parbe0, double parbe1, double parhe0, double parhe1) {
  fparhb0=parhb0;
  fparhb1=parhb1;
  fparbe0=parbe0;
  fparbe1=parbe1;
  fparhe0=parhe0;
  fparhe1=parhe1;
  fTimeSlew=tsParam;
  fTimeSlewBias=bias;
  fNegStrat=nStrat;
  fPedestalSubFxn_=pedSubFxn_;
}

constexpr float HcalDeterministicFit::landauFrac[];
// Landau function integrated in 1 ns intervals
//Landau pulse shape from https://indico.cern.ch/event/345283/contribution/3/material/slides/0.pdf
//Landau turn on by default at left edge of time slice 
// normalized to 1 on [0,10000]
void HcalDeterministicFit::getLandauFrac(float tStart, float tEnd, float &sum) const{

  if (std::abs(tStart-tEnd-25)<0.1) {
    sum=0;
    return;
  }
  sum= landauFrac[int(ceil(tStart+25))];
  return;
}

constexpr double HcalDeterministicFit::TS4par[];
constexpr double HcalDeterministicFit::TS5par[];
constexpr double HcalDeterministicFit::TS6par[];
