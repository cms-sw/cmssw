// #include "Utilities/Configuration/interface/Architecture.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/EtaCorrection.h"

// &&& PM: I presume this is going to go away.

#include <utility>
#include <cmath>
#include <iostream>

float
EtaCorrection::xEtaShift(const int& size, const float& pitch, 
			  const float& charatio, const float& alpha) const
{
  float eta = 0;
  float p1 = 0, p2 = 0, p3 = 0, p4 = 0;
  if (fabs(pitch-0.015) < 0.001 ) {
    if (size!= 1 && alpha < 1.53) {
      if (alpha < 1.45) {
	p1 = 0.18337E-02;
	p2 = -0.10775E-01;
	p3 = 0.21405E-01;
	p4 = -0.14590E-01;
      } else {
	p1 =0.44119E-03;
	p2 =-0.15944E-02;
	p3 =0.19076E-02;
	p4 =-0.11824E-02;
      }
      eta = (p1 + p2*charatio + p3*charatio*charatio 
	     + p4*pow(charatio,3));
    }
  } else if ( fabs(pitch-0.01) < 0.001 ) {
    if (size == 2 && alpha < 1.57) {
      if (alpha < 1.47) {
	p1 = 0.42949E-02;
	p2 = -0.19376E-01;
	p3 = 0.33451E-01;
	p4 = -0.23758E-01;
      } else {
	p1 = 0.17317E-02;
	p2 = -0.74960E-02;
	p3 = 0.11829E-01;
	p4 = -0.77922E-02;
      }
      eta = (p1 + p2*charatio + p3*charatio*charatio 
	     + p4*pow(charatio,3));
    }
  }
  return eta/pitch;  
}
float 
EtaCorrection::yEtaShift(const int& size, const float& pitch, 
			  const float& charatio, const float& beta) const
{
  float etashift = 0;
  float p1 = 0, p2 = 0, p3 = 0, p4 = 0;
  const float PI = 3.141593;
  /* eta correction for CMSIM
     if (fabs(pitch-0.015) < 0.001 ) {
     if (size == 2) {
      if ( fabs(PI/2-beta) >= 0.63) {
      etashift = -0.94900E-03 + 0.17031E-01*charatio - 
      0.45636E-01*charatio*charatio + 0.30617E-01*pow(charatio,3);
      }    
    } else if (size == 3) {
    if (fabs(PI/2-beta) >= 0.9){
    etashift = -0.52744E-02 +0.45230E-01*charatio
    -0.10433*charatio*charatio +0.69907E-01*pow(charatio,3);
    }
    } else if (size == 4) {
      if (fabs(PI/2-beta) >= 1.07){
      etashift = -0.44900E-03 +0.11220E-01*charatio
      -0.29842E-01*charatio*charatio +0.18076E-01*pow(charatio,3);
      }
      }
      }
  */
  // new eta correction for OSCAR_2_4_1
  if (fabs(pitch-0.015) < 0.001 ) {
    if (size == 2 ){
      if ( 0.37 < fabs(PI/2-beta) && fabs(PI/2-beta) < 0.63) {
	p1 = 0.81199E-03;
	p2 = -0.13107E-02;
	p3 = -0.11908E-02;
	p4 = 0.96596E-03;
      } else if ( 0.63 < fabs(PI/2-beta) ){
	/* gaussian fit	
	  p1 = 0.80863E-02;
	  p2 = -0.44447E-01;
	  p3 = 0.85808E-01;
	  p4 = -0.58170E-01;
	*/
	p1= 0.69865E-02;
	p2= -0.32312E-01;
	p3= 0.55552E-01;
	p4= -0.37662E-01;
      }
    } else if (size == 3) {
      if (fabs(PI/2-beta) <= 0.75) {
	p1 = 0.11717E-02;
	p2 = -0.57004E-02;
	p3 = 0.11013E-01;
	p4 = -0.82618E-02;
      } else if ( 0.75 < fabs(PI/2-beta) && fabs(PI/2-beta) <= 0.9 ) {
	p1 = 0.14907E-02;
	p2 = -0.61689E-02;
	p3 = 0.99599E-02;
	p4 = -0.70101E-02;
      } else if ( 0.9 < fabs(PI/2-beta)) {
	/* gaussian fit
	p1 = 0.13854E-01;
	p2 = -0.83617E-01;
	p3 = 0.16653;
	p4 = -0.10999;
	*/
	p1 = 0.11582E-01;
	p2 = -0.63337E-01;
	p3 = 0.11933;
	p4 = -0.78621E-01;
      }
    } else if (size == 4) {
      if (fabs(PI/2-beta) <= 0.98) {
	p1 = 0.24431E-02;
	p2 = -0.12762E-01;
	p3 = 0.22294E-01;
	p4 = -0.13710E-01;
      } else if (fabs(PI/2-beta)> 0.98 && fabs(PI/2-beta) <= 1.07){
	p1 = 0.16284E-02;
	p2 = -0.67447E-02;
	p3 = 0.99050E-02;
	p4 = -0.62142E-02;
      } else if ( fabs(PI/2-beta) > 1.07 ){
	/* gaussian fit
	p1 = 0.12823E-01;
	p2 = -0.74638E-01;
	p3 = 0.14592;
	p4 = -0.95965E-01;
	*/
	p1 = 0.91230E-02;
	p2 = -0.44437E-01;
	p3 = 0.78540E-01;
	p4 = -0.51896E-01;
      }
    }
    etashift = p1 + p2*charatio + p3*charatio*charatio + p4*pow(charatio,3);
  }
  return etashift/pitch;
}
