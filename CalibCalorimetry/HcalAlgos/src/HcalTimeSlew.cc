#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include <cmath>

static const double tzero[3]= {23.960177, 13.307784, 9.109694};
static const double slope[3] = {-3.178648,  -1.556668, -1.075824 };
static const double tmax[3] = {16.00, 10.00, 6.25 };
static const double cap = 6.0;
static const double tspar0[2] = {15.5, 12.2999};
static const double tspar1[2] = {-3.2,-2.19142};
static const double tspar2[2] = {32, 0};
static const double tspar0_siPM[2] = {3., 3.}; // 3ns delay for MC and DATA, recheck later for data
static const double tspar1_siPM[2] = {0, 0};
static const double tspar2_siPM[2] = {0, 0};

double HcalTimeSlew::delay(double fC, BiasSetting bias) {
  double rawDelay=tzero[bias]+slope[bias]*log(fC);
  return (rawDelay<0)?(0):((rawDelay>tmax[bias])?(tmax[bias]):(rawDelay));			   
}

double HcalTimeSlew::delay(double fC, ParaSource source, BiasSetting bias, double par0, double par1, double par2, bool isHPD) {

  if (source==TestStand) {
    return HcalTimeSlew::delay(fC, bias);
  }
  else if (source==InputPars) {
    return std::fmin(cap, par0 + par1*log(fC+par2));
  }
  else if (source==Data || source==MC){
    if(isHPD) return std::fmin(cap,tspar0[source-1]+tspar1[source-1]*log(fC+tspar2[source-1]));
    return cap+tspar0_siPM[source-1];
  }
  return 0;
}
