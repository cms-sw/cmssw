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

// Landau function integrated in 1 ns intervals
//Landau pulse shape from https://indico.cern.ch/event/345283/contribution/3/material/slides/0.pdf
//Landau turn on by default at left edge of time slice 
// normalized to 1 on [0,10000]
void HLTv2::getLandauFrac(Float_t tStart, Float_t tEnd, Float_t &sum) const{

   Float_t landauFrac[125] = {0, 7.6377e-05, 0.000418655, 0.00153692, 0.00436844, 0.0102076, 0.0204177, 0.0360559, 0.057596, 0.0848493, 0.117069, 0.153152, 0.191858, 0.23198, 0.272461, 0.312438, 0.351262, 0.388476, 0.423788, 0.457036, 0.488159, 0.517167, 0.54412, 0.569112, 0.592254, 0.613668, 0.633402, 0.651391, 0.667242, 0.680131, 0.688868, 0.692188, 0.689122, 0.67928, 0.662924, 0.64087, 0.614282, 0.584457, 0.552651, 0.51997, 0.487317, 0.455378, 0.424647, 0.395445, 0.367963, 0.342288, 0.318433, 0.29636, 0.275994, 0.257243, 0.24, 0.224155, 0.2096, 0.196227, 0.183937, 0.172635, 0.162232, 0.15265, 0.143813, 0.135656, 0.128117, 0.12114, 0.114677, 0.108681, 0.103113, 0.0979354, 0.0931145, 0.0886206, 0.0844264, 0.0805074, 0.0768411, 0.0734075, 0.0701881, 0.0671664, 0.0643271, 0.0616564, 0.0591418, 0.0567718, 0.054536, 0.0524247, 0.0504292, 0.0485414, 0.046754, 0.0450602, 0.0434538, 0.041929, 0.0404806, 0.0391037, 0.0377937, 0.0365465, 0.0353583, 0.0342255, 0.0331447, 0.032113, 0.0311274, 0.0301854, 0.0292843, 0.0284221, 0.0275964, 0.0268053, 0.0253052, 0.0238536, 0.0224483, 0.0210872, 0.0197684, 0.0184899, 0.01725, 0.0160471, 0.0148795, 0.0137457, 0.0126445, 0.0115743, 0.0105341, 0.00952249, 0.00853844, 0.00758086, 0.00664871,0.00574103, 0.00485689, 0.00399541, 0.00315576, 0.00233713, 0.00153878, 0.000759962, 0 };

  if (abs(tStart-tEnd-25)<0.1) {
    cout << abs(tStart-tEnd) << "???" << endl;
    sum=0;
    return;
  }
  sum=landauFrac[int(ceil(tStart+25))];
  return;
}

void HLTv2::PulseFraction(Double_t fC, Double_t *TS46) const{

  //static Double_t TS3par[3] = {0.44, -18.6, 5.136}; //Gaussian parameters: norm, mean, sigma for the TS3 fraction                    
  static Double_t TS4par[3] = {0.71, -5.17, 12.23}; //Gaussian parameters: norm, mean, sigma for the TS4 fraction                      
  static Double_t TS5par[3] = {0.258, 0.0178, 4.786e-4}; // pol2 parameters for the TS5 fraction                                       
  static Double_t TS6par[4] = {0.06391, 0.002737, 8.396e-05, 1.475e-06};// pol3 parameters for the TS6 fraction                        

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
