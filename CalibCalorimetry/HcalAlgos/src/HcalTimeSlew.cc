#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include <cmath>

static const double tzero[3]= {23.960177, 13.307784, 9.109694};
static const double slope[3] = {-3.178648,  -1.556668, -1.075824 };
static const double tmax[3] = {16.00, 10.00, 6.25 };

double HcalTimeSlew::delay(double fC, BiasSetting bias) {
  double rawDelay=tzero[bias]+slope[bias]*log(fC);
  return (rawDelay<0)?(0):((rawDelay>tmax[bias])?(tmax[bias]):(rawDelay));			   
}

double HcalTimeSlew::delay(double fC, ParaSource source, BiasSetting bias) {

  if (source==TestStand) {
    return HcalTimeSlew::delay(fC, bias);
  }
  else if (source==Data) {
    //from john 2/20 talk: indico.cern.ch/event/375365/contribution/9/material/slides/5.pdf
    return 13.98-3.20*log(fC+32)-2.82965+10;
  }
  else if (source==MC) {
    // from Xinmei
    //return 10.491-2.25495*log(fC+7.95067);
    
    //FCN=1075.01 FROM MIGRAD    STATUS=CONVERGED      34 CALLS          35 TOTAL
    //EDM=1.84101e-19    STRATEGY= 1      ERROR MATRIX ACCURATE 
    //EXT PARAMETER                                   STEP         FIRST   
    //NO.   NAME      VALUE            ERROR          SIZE      DERIVATIVE 
    //1  p0           9.27638e+00   3.45586e-02   1.17592e-04   8.21769e-08
    //2  p1          -2.05585e+00   1.10061e-02   3.74505e-05   2.58031e-07

    return std::min(6.0,9.27638-2.05585*log(fC));
    
  }  
  return 0;
}
