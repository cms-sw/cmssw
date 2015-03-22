#include <iostream>
#include <cmath>
#include <climits>
#include "RecoLocalCalo/HcalRecAlgos/interface/HLTv2.h"

using namespace std;

HLTv2::HLTv2() {
}

HLTv2::~HLTv2() { 
}

void HLTv2::Init(HcalTimeSlew::ParaSource tsParam, HcalTimeSlew::BiasSetting bias, NegStrategy nStrat, PedestalSub pedSubFxn_) {
  fTimeSlew=tsParam;
  fTimeSlewBias=bias;
  fNegStrat=nStrat;
  fPedestalSubFxn_=pedSubFxn_;
}
void HLTv2::apply(const CaloSamples & cs, const std::vector<int> & capidvec, const HcalCalibrations & calibs, std::vector<double> & HLTOutput) const {
//void HLTv2::apply(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal, std::vector<double> & HLTOutput) const {

  std::vector<double> corrCharge;
  std::vector<double> inputCharge;
  std::vector<double> inputPedestal;
  const unsigned int cssize = cs.size();
  // initialize arrays to be zero
 // double tstrig = 0; // in fC
 // double tsTOTen = 0; // in GeV
  double gainCorr = 0;
  for(unsigned int ip=0; ip<cssize; ++ip){
  //  std::cout << "assigning vectors " << std::endl;
    if( ip >= (unsigned) 10 ) continue; // Too many samples than what we wanna fit (10 is enough...) -> skip them
    const int capid = capidvec[ip];
    double charge = cs[ip];
    double ped = calibs.pedestal(capid);
    double gain = calibs.respcorrgain(capid);
    gainCorr = gain;
    inputCharge.push_back(charge);
    inputPedestal.push_back(ped);
    //corrCharge.push_back(charge - ped); // Using Pedestal-subtracted values
  }                   
 
  //std::cout << "at ped sub" << std::endl; 
  fPedestalSubFxn_.Calculate(inputCharge, inputPedestal, corrCharge);
  
  Float_t tsShift3=HcalTimeSlew::delay(inputCharge[3],HcalTimeSlew::MC,fTimeSlewBias); 
  Float_t tsShift4=HcalTimeSlew::delay(inputCharge[4],HcalTimeSlew::MC,fTimeSlewBias); 
  Float_t tsShift5=HcalTimeSlew::delay(inputCharge[5],HcalTimeSlew::MC,fTimeSlewBias); 

  // pulses are delayed by tshift. e.g. tshift = 10 means pulse is 10 seconds later
  // landau frac moves limits of integration to the left by tsshift, which is equivalent
  // to moving the pulse to the right
  Float_t i3=0;
  getLandauFrac(-tsShift3,-tsShift3+25,i3);
  Float_t n3=0;
  getLandauFrac(-tsShift3+25,-tsShift3+50,n3);
  Float_t nn3=0;
  getLandauFrac(-tsShift3+50,-tsShift3+75,nn3);

  Float_t i4=0;
  getLandauFrac(-tsShift4,-tsShift4+25,i4);
  Float_t n4=0;
  getLandauFrac(-tsShift4+25,-tsShift4+50,n4);

  Float_t i5=0;
  getLandauFrac(-tsShift5,-tsShift5+25,i5);
  Float_t n5=0;
  getLandauFrac(-tsShift5+25,-tsShift5+50,n5);

  Float_t ch3=corrCharge[3]/i3;
  Float_t ch4=(i3*corrCharge[4]-n3*corrCharge[3])/(i3*i4);
  Float_t ch5=(n3*n4*corrCharge[3]-i4*nn3*corrCharge[3]-i3*n4*corrCharge[4]+i3*i4*corrCharge[5])/(i3*i4*i5);

  if (ch3<-3 && fNegStrat==HLTv2::ReqPos) {
    ch3=-3;
    ch4=corrCharge[4]/i4;
    ch5=(i4*corrCharge[5]-n4*corrCharge[4])/(i4*i5);
  }
  
  if (ch5<-3 && fNegStrat==HLTv2::ReqPos) {
    ch4=ch4+(ch5+3);
    ch5=-3;
  }

  if (fNegStrat==HLTv2::FromGreg) {
    if (ch3<-3) {
      ch3=-3;
      ch4=corrCharge[4]/i4;
      ch5=(i4*corrCharge[5]-n4*corrCharge[4])/(i4*i5);
    }
    if (ch5<-3 && ch4>15) {
      double ratio = (corrCharge[4]-ch3*i3)/(corrCharge[5]+3*i5);
      if (ratio < 5 && ratio > 0.5) {
	double invG = -13.11+11.29*TMath::Sqrt(2*TMath::Log(5.133/ratio));
	Float_t iG=0;
	getLandauFrac(-invG,-invG+25,iG);
	ch4=(corrCharge[4]-ch3*n3)/(iG);
	ch5=-3;
	tsShift4=invG;
      }
    }
  }
  
  if (ch3<1) {// && (fNegStrat==HLTv2::ReqPos || fNegStrat==HLTv2::FromGreg)) {
    ch3=0;
  }
  if (ch4<1) {// && (fNegStrat==HLTv2::ReqPos || fNegStrat==HLTv2::FromGreg)) {
    ch4=0;
  }
  if (ch5<1) {// && (fNegStrat==HLTv2::ReqPos || fNegStrat==HLTv2::FromGreg)) {
    ch5=0;
  }
  
  // Print out 
  HLTOutput.clear();
  HLTOutput.push_back(ch4*gainCorr);// amplitude 
  HLTOutput.push_back(tsShift4); // time shift of in-time pulse
  HLTOutput.push_back(ch5); // whatever

}

void HLTv2::applyXM(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal, std::vector<double> & HLTOutput) const {

  std::vector<double> corrCharge;
  fPedestalSubFxn_.Calculate(inputCharge, inputPedestal, corrCharge);

  double TS35[3];
  double TS46[3];
  double TS57[3];
  PulseFraction(inputCharge[3], TS35);
  PulseFraction(inputCharge[4], TS46);
  PulseFraction(inputCharge[5], TS57);

  double a3[3] = {TS35[0], TS35[1], TS35[2]};
  double b3[3] = {0., TS46[0], TS46[1]};
  double c3[3] = {0., 0., TS57[0]};
  double d3[3] = {corrCharge[3], corrCharge[4], corrCharge[5]};

  double deno3 = Det3(a3, b3, c3);

  double A3 = Det3(d3, b3, c3) / deno3;
  double A4 = Det3(a3, d3, c3) / deno3;
  double A5 = Det3(a3, b3, d3) / deno3;
  HLTOutput.clear();
  HLTOutput.push_back(A3);
  HLTOutput.push_back(A4);
  HLTOutput.push_back(A5);

}

constexpr Float_t HLTv2::landauFrac[];
// Landau function integrated in 1 ns intervals
//Landau pulse shape from https://indico.cern.ch/event/345283/contribution/3/material/slides/0.pdf
//Landau turn on by default at left edge of time slice 
// normalized to 1 on [0,10000]
void HLTv2::getLandauFrac(Float_t tStart, Float_t tEnd, Float_t &sum) const{

  if (abs(tStart-tEnd-25)<0.1) {
    cout << abs(tStart-tEnd) << "???" << endl;
    sum=0;
    return;
  }
  sum= landauFrac[int(ceil(tStart+25))];
  return;
}

constexpr Double_t HLTv2::TS4par[];
constexpr Double_t HLTv2::TS5par[];
constexpr Double_t HLTv2::TS6par[];

void HLTv2::PulseFraction(Double_t fC, Double_t *TS46) const{

  Double_t tslew = -HcalTimeSlew::delay(fC,HcalTimeSlew::MC,fTimeSlewBias);

  TS46[0] = TS4par[0] * TMath::Gaus(tslew,TS4par[1],TS4par[2]); // fraction of pulse in the TS4          
  TS46[1] = TS5par[0] + TS5par[1]*tslew + TS5par[2]*tslew*tslew; // fraction of pulse in the T5S
  TS46[2] = TS6par[0] + TS6par[1]*tslew + TS6par[2]*tslew*tslew + TS6par[3]*tslew*tslew*tslew; //fraction of pulse in the TS6

  return;
}

double HLTv2::Det2(double *b, double *c) const{
  return b[1]*c[2]-b[2]*c[1];
}

double HLTv2::Det3(double *a, double *b, double *c) const{
  return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]);
}
