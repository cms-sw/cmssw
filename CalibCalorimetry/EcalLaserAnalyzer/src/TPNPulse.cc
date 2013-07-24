/* 
 *  \class TPNPulse
 *
 *  $Date: 2012/02/09 10:08:10 $
 *  \author: Julie Malcles - CEA/Saclay
 */

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TPNPulse.h>

#include <TMath.h>
#include <iostream>
#include <cassert>
using namespace std;

//ClassImp(TPNPulse)


// Default Constructor...
TPNPulse::TPNPulse()
{
  init(50,6);
}

// Constructor...
TPNPulse::TPNPulse( int nsamples, int presample )
{
  init( nsamples,  presample);
}

// Destructor
TPNPulse::~TPNPulse()
{
}

void TPNPulse::init(int nsamples, int presample )
{
  _nsamples=50;
  assert(nsamples==_nsamples);
  assert(presample!=0);
  adc_ = new double[50];  

  _presample=presample;
  
  for(int i=0;i<_nsamples;i++){
    adc_[i]=0.0;
  }

  adcMax_=0;
  iadcMax_=0;
  pedestal_=0;
  
  isMaxFound_=false;
  isPedCalc_=false;
}

bool TPNPulse::setPulse(double *adc){

  bool done=false;
  adc_=adc;
  done=true;
  isMaxFound_=false;
  isPedCalc_=false;
  return done;
}
double TPNPulse::getMax(){

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

int TPNPulse::getMaxSample(){
  if(!isMaxFound_) getMax();
  return iadcMax_;

}

double TPNPulse::getPedestal(){
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

double* TPNPulse::getAdcWithoutPedestal(){
  
  double ped;
  if(!isPedCalc_) ped=getPedestal();
  else ped=pedestal_;
  
  double *adcNoPed= new double[50];
  for (int i=0;i<_nsamples;i++){
    adcNoPed[i]=adc_[i]-ped;
  }
  return adcNoPed;  
}

void TPNPulse::setPresamples(int presample){
  isPedCalc_=false;
  _presample=presample;
}
