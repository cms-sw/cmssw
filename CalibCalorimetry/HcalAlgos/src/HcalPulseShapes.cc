#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include <cmath>

HcalPulseShapes::HcalPulseShapes() {
  computeHPDShape(hpdShape_);
  computeHFShape(hfShape_);
}


void HcalPulseShapes::computeHPDShape(HcalPulseShapes::Shape& sh)
{

  // pulse shape time constants in ns
  const float ts1  = 8.;          // scintillation time constants : 1,2,3
  const float ts2  = 10.;           
  const float ts3  = 29.3;         
  const float thpd = 4.;          // HPD current collection drift time
  const float tpre = 9.;          // preamp time constant (refit on TB04 data)
  
  const float wd1 = 2.;           // relative weights of decay exponents 
  const float wd2 = 0.7;
  const float wd3 = 1.;
  
  // pulse shape componnts over a range of time 0 ns to 255 ns in 1 ns steps
  int nbin = 256;
  sh.setNBin(nbin);
  std::vector<float> ntmp(nbin,0.0);  // zeroing output pulse shape
  std::vector<float> nth(nbin,0.0);   // zeroing HPD drift shape
  std::vector<float> ntp(nbin,0.0);   // zeroing Binkley preamp shape
  std::vector<float> ntd(nbin,0.0);   // zeroing Scintillator decay shape

  int i,j,k;
  float norm;

  // HPD starts at I and rises to 2I in thpd of time
  norm=0.0;
  for(j=0;j<thpd && j<nbin;j++){
    nth[j] = 1.0 + ((float)j)/thpd;
    norm += nth[j];
  }
  // normalize integrated current to 1.0
  for(j=0;j<thpd && j<nbin;j++){
    nth[j] /= norm;
  }
  
  // Binkley shape over 6 time constants
  norm=0.0;
  for(j=0;j<6*tpre && j<nbin;j++){
    ntp[j] = ((float)j)*exp(-((float)(j*j))/(tpre*tpre));
    norm += ntp[j];
  }
  // normalize pulse area to 1.0
  for(j=0;j<6*tpre && j<nbin;j++){
    ntp[j] /= norm;
  }

// ignore stochastic variation of photoelectron emission
// <...>

// effective tile plus wave-length shifter decay time over 4 time constants
  int tmax = 6 * (int)ts3;
 
  norm=0.0;
  for(j=0;j<tmax && j<nbin;j++){
    ntd[j] = wd1 * exp(-((float)j)/ts1) + 
      wd2 * exp(-((float)j)/ts2) + 
      wd3 * exp(-((float)j)/ts3) ; 
    norm += ntd[j];
  }
  // normalize pulse area to 1.0
  for(j=0;j<tmax && j<nbin;j++){
    ntd[j] /= norm;
  }
  
  int t1,t2,t3,t4;
  for(i=0;i<tmax && i<nbin;i++){
    t1 = i;
    //    t2 = t1 + top*rand;
    // ignoring jitter from optical path length
    t2 = t1;
    for(j=0;j<thpd && j<nbin;j++){
      t3 = t2 + j;
      for(k=0;k<4*tpre && k<nbin;k++){       // here "4" is set deliberately,
 t4 = t3 + k;                         // as in test fortran toy MC ...
 if(t4<nbin){                         
   int ntb=t4;                        
   ntmp[ntb] += ntd[i]*nth[j]*ntp[k];
	}
      }
    }
  }
  
  // normalize for 1 GeV pulse height
  norm = 0.;
  for(i=0;i<nbin;i++){
    norm += ntmp[i];
  }

  //cout << " Convoluted SHAPE ==============  " << endl;
  for(i=0; i<nbin; i++){
    ntmp[i] /= norm;
    //  cout << " shape " << i << " = " << ntmp[i] << endl;   
  }

  for(i=0; i<nbin; i++){
    sh.setShapeBin(i,ntmp[i]);
  }
}

void HcalPulseShapes::computeHFShape(HcalPulseShapes::Shape& sh) {
  // first create pulse shape over a range of time 0 ns to 255 ns in 1 ns steps
  int nbin = 256;
  sh.setNBin(nbin);
  std::vector<float> ntmp(nbin,0.0);  // 

  const float k0=0.7956; // shape parameters
  const float p2=1.355;
  const float p4=2.327;
  const float p1=4.3;    // position parameter

  float norm = 0.0;

  for(int j = 0; j < 25 && j < nbin; ++j){

    float r0 = j-p1;
    float sigma0 = (r0<0) ? p2 : p2*p4;
    r0 /= sigma0;
    if(r0 < k0) ntmp[j] = exp(-0.5*r0*r0);
    else ntmp[j] = exp(0.5*k0*k0-k0*r0);
    norm += ntmp[j];
  }
  // normalize pulse area to 1.0
  for(int j = 0; j < 25 && j < nbin; ++j){
    ntmp[j] /= norm;
    sh.setShapeBin(j,ntmp[j]);
  }
}

HcalPulseShapes::Shape::Shape() {
  nbin_=0;
  tpeak_=0;
}

void HcalPulseShapes::Shape::setNBin(int n) {
  nbin_=n;
  shape_=std::vector<float>(n,0.0f);
}

void HcalPulseShapes::Shape::setShapeBin(int i, float f) {
  if (i>=0 && i<nbin_) shape_[i]=f;
}

float HcalPulseShapes::Shape::operator()(double t) const {
  // shape is in 1 ns steps
  return at(t);
}

float HcalPulseShapes::Shape::at(double t) const {
  // shape is in 1 ns steps
  int i=(int)(t+0.5);
  float rv=0;
  if (i>=0 && i<nbin_) rv=shape_[i];
  return rv;
}

float HcalPulseShapes::Shape::integrate(double t1, double t2) const {
  static const float int_delta_ns = 0.05f; 
  double intval = 0.0;

  for (double t = t1; t < t2; t+= int_delta_ns) {
    float loedge = at(t);
    float hiedge = at(t+int_delta_ns);
    intval += (loedge+hiedge)*int_delta_ns/2.0;
  }
  return (float)intval;
}
