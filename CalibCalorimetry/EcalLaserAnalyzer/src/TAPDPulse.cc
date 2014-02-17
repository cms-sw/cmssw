/* 
 *  \class TAPDPulse
 *
 *  $Date: 2012/02/09 10:08:10 $
 *  \author: Julie Malcles - CEA/Saclay
 */

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TAPDPulse.h>
#include <TMath.h>
#include <iostream>
#include <cassert>
using namespace std;

//ClassImp(TAPDPulse)


// Default Constructor...
TAPDPulse::TAPDPulse()
{
  init(10,3,1,2,2,9,3,8,0.4,0.95,0.8);
}

// Constructor...
TAPDPulse::TAPDPulse( int nsamples, int presample, int firstsample, int lastsample, int timingcutlow, int timingcuthigh, int timingquallow, int timingqualhigh, double ratiomincutlow, double ratiomincuthigh, double ratiomaxcutlow)
{
  init( nsamples,  presample,  firstsample,  lastsample,  timingcutlow, timingcuthigh,  timingquallow,  timingqualhigh,ratiomincutlow,ratiomincuthigh, ratiomaxcutlow );
}

// Destructor
TAPDPulse::~TAPDPulse()
{
}

void TAPDPulse::init(int nsamples, int presample, int firstsample, int lastsample, int timingcutlow, int timingcuthigh, int timingquallow, int timingqualhigh, double ratiomincutlow, double ratiomincuthigh, double ratiomaxcutlow)
{
  _nsamples=10;
  assert(nsamples==_nsamples);
  assert(presample!=0);
  adc_ = new double[10];  

  _presample=presample;
  _firstsample=firstsample;
  _lastsample=lastsample;
  
  _timingcutlow=timingcutlow;
  _timingcuthigh=timingcuthigh;
  _timingquallow=timingquallow;
  _timingqualhigh=timingqualhigh;
  _ratiomincutlow=ratiomincutlow;
  _ratiomincuthigh=ratiomincuthigh;
  _ratiomaxcutlow=ratiomaxcutlow;

  for(int i=0;i<_nsamples;i++){
    adc_[i]=0.0;
  }

  adcMax_=0;
  iadcMax_=0;
  pedestal_=0;
  
  isMaxFound_=false;
  isPedCalc_=false;
}

bool TAPDPulse::setPulse(double *adc){

  bool done=false;
  adc_=adc;
  done=true;
  isMaxFound_=false;
  isPedCalc_=false;
  return done;
}
double TAPDPulse::getMax(){

  if(isMaxFound_) return adcMax_; 

  int iadcmax=0;
  double adcmax=0.0;
  for(int i=0;i<_nsamples;i++){
    if(adc_[i]>adcmax){
      iadcmax=i;
      adcmax=adc_[i];
    }
  }
  iadcMax_=iadcmax;
  adcMax_=adcmax;
  return adcMax_;  
}

int TAPDPulse::getMaxSample(){
  if(!isMaxFound_) getMax();
  return iadcMax_;

}
double TAPDPulse::getDelta(int n1, int n2){

  assert (n1<_nsamples && n1>=0);
  assert (n2<_nsamples && n2>=0);
  
  double delta=adc_[n1]-adc_[n2];
  return delta;
}
double TAPDPulse::getRatio(int n1, int n2){
  
  assert (n1<_nsamples && n1>=0);
  assert (n2<_nsamples && n2>=0);

  double ped=0;
  if(isPedCalc_)ped=pedestal_; 
  else ped=adc_[0];
  
  double ratio=(adc_[n1]-ped)/(adc_[n2]-ped);
  return ratio; 
}

bool TAPDPulse::isTimingOK(){

  bool ok=true;
  if(!isMaxFound_) getMax();
  if(iadcMax_<=_timingcutlow || iadcMax_>=_timingcuthigh) ok=false;
  return ok;
}
bool TAPDPulse::isTimingQualOK(){

  bool ok=true;
  if(!isMaxFound_) getMax();
  if(iadcMax_<=_timingquallow || iadcMax_>=_timingqualhigh) ok=false;
  return ok;
}

bool TAPDPulse::areFitSamplesOK(){
  
  bool ok=true;
  if(!isMaxFound_) getMax();
  if ((iadcMax_-_firstsample)<_presample || (iadcMax_+_lastsample)>_nsamples-1) ok=false;
  return ok;
  
}
bool TAPDPulse::isPulseOK(){

  bool okSamples=areFitSamplesOK();
  bool okTiming=isTimingOK();
  bool okPulse=arePulseRatioOK();

  bool ok=(okSamples && okTiming && okPulse);

  return ok;
}
bool TAPDPulse::arePulseRatioOK(){

  bool ok=true;

  if(!isMaxFound_) getMax();
  if(iadcMax_<1 || iadcMax_>=_nsamples-1) return false;
  
  double ratioNm1=getRatio(iadcMax_-1,iadcMax_);
  double ratioNp1=getRatio(iadcMax_+1,iadcMax_);
  double ratioMax=TMath::Max(ratioNm1,ratioNp1);
  double ratioMin=TMath::Min(ratioNm1,ratioNp1);
  
  if(ratioMax<_ratiomaxcutlow) ok=false;
  if(ratioMin<_ratiomincutlow || ratioMin>_ratiomincuthigh) ok=false;

  return ok;

}
bool TAPDPulse::isPulseRatioMaxOK(){

  bool ok=true;

  if(!isMaxFound_) getMax();
  if(iadcMax_<1 || iadcMax_>=_nsamples-1) return false;
  
  double ratioNm1=getRatio(iadcMax_-1,iadcMax_);
  double ratioNp1=getRatio(iadcMax_+1,iadcMax_);
  double ratioMax=TMath::Max(ratioNm1,ratioNp1);
  
  if(ratioMax<_ratiomaxcutlow) ok=false;
  return ok;

}
bool TAPDPulse::isPulseRatioMinOK(){

  bool ok=true;
  
  if(!isMaxFound_) getMax();
  if(iadcMax_<1 || iadcMax_>=_nsamples-1) return false;
  
  double ratioNm1=getRatio(iadcMax_-1,iadcMax_);
  double ratioNp1=getRatio(iadcMax_+1,iadcMax_);
  double ratioMin=TMath::Min(ratioNm1,ratioNp1);
  
  if(ratioMin<_ratiomincutlow || ratioMin>_ratiomincuthigh) ok=false;
  return ok;  
}

double TAPDPulse::getPedestal(){
  if(isPedCalc_) return pedestal_;
  double ped=0;
  for(int i=0;i<_presample;i++){
    ped+=adc_[i];
  }
  ped/=double(_presample);
  pedestal_=ped;
  isPedCalc_=true;
  return pedestal_;
}

double* TAPDPulse::getAdcWithoutPedestal(){
  
  double ped;
  if(!isPedCalc_) ped=getPedestal();
  else ped=pedestal_;
  
  double *adcNoPed= new double[10];
  for (int i=0;i<_nsamples;i++){
    adcNoPed[i]=adc_[i]-ped;
  }
  return adcNoPed;  
}

void TAPDPulse::setPresamples(int presample){
  isPedCalc_=false;
  _presample=presample;
}
