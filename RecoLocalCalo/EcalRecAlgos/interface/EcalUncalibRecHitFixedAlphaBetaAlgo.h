#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitFixedAlphaBetaAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitFixedAlphaBetaAlgo_HH

/** \class EcalUncalibRecHitFixedAlphaBetaAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using an analytical function fit, with the pulse parameters alpha and beta fixed.
  *  It follows a fast fit algorithms devolped on test beam data by P. Jarry
  *  If the pedestal is not given, it is calculated from the first 2 pre-samples; 
  *  FIXME: conversion gainID (1,2,3) with gain factor (12,6,1) is hardcoded here !  
  *
  *  \author A.Ghezzi
  */

#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Matrix/SymMatrix.h"
//#include "CLHEP/Matrix/Matrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <string>

//#include "TROOT.h"
//#include "TMinuit.h"
//#include "TGraph.h"
//#include "TF1.h"
//#include "TMatrixD.h"
//#include "TVectorD.h"

template<class C> class EcalUncalibRecHitFixedAlphaBetaAlgo : public EcalUncalibRecHitRecAbsAlgo<C>
{


 private:

  double fAlpha_;//parameter of the shape
  double fBeta_;//parameter of the shape
  double fAmp_max_;// peak amplitude 
  double fTim_max_ ;// time of the peak (in 25ns units)
  double fPed_max_;// pedestal value
  double alfabeta_; 


  int fNb_iter_;
  int  fNum_samp_bef_max_ ;
  int fNum_samp_after_max_;
  
  float fSigma_ped; 
  double un_sur_sigma;
  //temporary solution for different alpha and beta
  float alpha_table_[36][1701];
  float beta_table_[36][1701];

  double pulseShapeFunction(double t);
  float PerformAnalyticFit(double* samples, int max_sample);
  void InitFitParameters(double* samples, int max_sample);
  HepSymMatrix DM1_ ; HepVector temp_;
   public:

  EcalUncalibRecHitFixedAlphaBetaAlgo<C>():fAlpha_(0.),fBeta_(0.),fAmp_max_(-1.),fTim_max_(-1),fPed_max_(0),alfabeta_(0),fNb_iter_(4),fNum_samp_bef_max_(1),fNum_samp_after_max_(3),fSigma_ped(1.1),DM1_(3),temp_(3){
    un_sur_sigma = 1./double(fSigma_ped) ;
    for (int i=0;i<36;i++){
      for(int j=0;j<1701;j++){
	alpha_table_[i][j] = 1.2 ;
	beta_table_[i][j]  =  1.7 ;
      }
    }
  }
  EcalUncalibRecHitFixedAlphaBetaAlgo<C>(int n_iter, int n_bef_max =1, int n_aft_max =3, float sigma_ped = 1.1):fAlpha_(0.),fBeta_(0.),fAmp_max_(-1.),fTim_max_(-1),fPed_max_(0),alfabeta_(0),DM1_(3),temp_(3){
    
   fNb_iter_ = n_iter ;
   fNum_samp_bef_max_   = n_bef_max ;
   fNum_samp_after_max_ = n_aft_max ;
   fSigma_ped = sigma_ped;
   un_sur_sigma = 1./double(fSigma_ped) ;
   for (int i=0;i<36;i++){
     for(int j=0;j<1701;j++){
       alpha_table_[i][j] = 1.2 ;
       beta_table_[i][j]  =  1.7 ;
     }
   }
   
  };

  virtual ~EcalUncalibRecHitFixedAlphaBetaAlgo<C>() { };
  virtual EcalUncalibratedRecHit makeRecHit(const C& dataFrame, const std::vector<double>& pedestals,
                                            const std::vector<double>& gainRatios,
					    const std::vector<HepMatrix>& weights,
                                            const std::vector<HepSymMatrix>& chi2Matrix);
  void SetAlphaBeta( double alpha, double beta);
  
};



  /// Compute parameters
template<class C> EcalUncalibratedRecHit  EcalUncalibRecHitFixedAlphaBetaAlgo<C>::makeRecHit(const C& dataFrame, const std::vector<double>& pedestals, const std::vector<double>& gainRatios, const std::vector<HepMatrix>& weights, const std::vector<HepSymMatrix>& chi2Matrix) {
  double chi2_(-1.);
  
  //  double Gain12Equivalent[4]={0,1,2,12};
  double frame[C::MAXSAMPLES];// will contain the ADC values
  double pedestal =0;
  
  int gainId0 = dataFrame.sample(0).gainId();
  int iGainSwitch = 0;
  int GainId= 0;
  double maxsample(-1);
  int imax(-1);
  bool external_pede = false;
  // Get time samples checking for Gain Switch and pedestals
  if(pedestals.size()==3){
    external_pede = true;
    pedestal = pedestals[0];
      for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
	//create frame in adc gain 12 equivalent
	GainId = dataFrame.sample(iSample).gainId();
	// FIX-ME: warning: the vector pedestal is supposed to have in the order G12, G6 and G1
	frame[iSample] = (double(dataFrame.sample(iSample).adc())-pedestals[GainId-1])*gainRatios[GainId-1];
	//Gain12Equivalent[GainId];
	if (GainId > gainId0) iGainSwitch = 1;
	if( frame[iSample]>maxsample ) {
          maxsample = frame[iSample];
          imax = iSample;
	}
      }
  }
  else {// pedestal from pre-sample
    external_pede = false;
    pedestal = (double(dataFrame.sample(0).adc()) + double(dataFrame.sample(1).adc()))/2.;
    for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
      //create frame in adc gain 12 equivalent
      GainId = dataFrame.sample(iSample).gainId();
      //no gain switch forseen if there is no external pedestal
      frame[iSample] = double(dataFrame.sample(iSample).adc())-pedestal ;
      if (GainId > gainId0) iGainSwitch = 1;
      if( frame[iSample]>maxsample ) {
	maxsample = frame[iSample];
	imax = iSample;
      }
    } 
  }

  if(iGainSwitch==1 && external_pede==false){
    return EcalUncalibratedRecHit( dataFrame.id(), -1., -100., -1. , -1.);
  }
  
  InitFitParameters(frame, imax);
  chi2_ = PerformAnalyticFit(frame,imax);

  /*    std::cout << "separate fits\nA: " << fAmp_max_  << ", ResidualPed: " <<  fPed_max_
              <<", pedestal: "<<pedestal << ", tPeak " << fTim_max_ << std::endl;
  */
  return EcalUncalibratedRecHit( dataFrame.id(),fAmp_max_, pedestal+fPed_max_, fTim_max_ , chi2_ );
}

template<class C> double EcalUncalibRecHitFixedAlphaBetaAlgo<C>::pulseShapeFunction(double t){
  if( alfabeta_ <= 0 ) return((double)0.);
  double dtsbeta,variable,puiss;
  double dt = t-fTim_max_ ;
  if(dt > -alfabeta_)  {
    dtsbeta=dt/fBeta_ ;
    variable=1.+dt/alfabeta_ ;
    puiss=pow(variable,fAlpha_);
    return fAmp_max_*puiss*exp(-dtsbeta)+fPed_max_ ;
  }
  return  fPed_max_ ;
}

template<class C> void  EcalUncalibRecHitFixedAlphaBetaAlgo<C>::InitFitParameters(double* samples, int max_sample){
  // in a first attempt just use the value of the maximum sample 
  fAmp_max_ = samples[max_sample];
  fTim_max_ = max_sample;

  // use a 2nd polynomial around maximum
  if( max_sample < 1){return;}
  //y=a*(x-xM)^2+b*(x-xM)+c
  float a = float(samples[max_sample-1]+samples[max_sample+1]-2*samples[max_sample])/2.;
  if(a==0){return;}
  float b = float(samples[max_sample+1]-samples[max_sample-1])/2.;
  
  fTim_max_ = max_sample - b/(2*a);
  fAmp_max_ =  samples[max_sample] - b*b/(4*a); 
  fPed_max_ = 0;
} 

template<class C> float EcalUncalibRecHitFixedAlphaBetaAlgo<C>::PerformAnalyticFit(double* samples, int max_sample){
  
  //int fValue_tim_max = max_sample;  
  //! fit electronic function from simulation
  //! parameters fAlpha_ and fBeta_ are fixed and fit is providing the 3 following parameters
  //! the maximum amplitude ( fAmp_max_ ) 
  //! the time of the maximum  ( fTim_max_)
  //| the pedestal (fPed_max_) 	
  
  double chi2=-1 , db[3] ;
  

  //HepSymMatrix DM1(3) ; HepVector temp(3) ;

  int num_fit_min =(int)(max_sample - fNum_samp_bef_max_ ) ;
  int num_fit_max =(int)(max_sample + fNum_samp_after_max_) ;

  if (num_fit_min<0) num_fit_min = 0 ; 
  //if (num_fit_max>=fNsamples-1) num_fit_max = fNsamples-2 ;
  if (num_fit_max>= C::MAXSAMPLES ) {num_fit_max = C::MAXSAMPLES-1 ;}

  if(fAmp_max_ < 8.) {
    LogDebug("EcalUncalibRecHitFixedAlphaBetaAlgo")<<"amplitude less than 8 ADC counts, no fit performed"; return -1;
    return -1;
  }

  double func,delta ;
  double variation_func_max = 0. ;double variation_tim_max = 0. ; double variation_ped_max = 0. ;
  //!          Loop on iterations
  for (int iter=0 ; iter < fNb_iter_ ; iter ++) {
    //!          initialization inside iteration loop !
    chi2 = 0. ; //PROD.Zero() ;  DM1.Zero() ;

     for(int i1=0 ; i1<3 ; i1++) {
       temp_[i1]=0;
	for(int j1=i1 ; j1<3 ; j1++) { 
	  DM1_.fast(j1+1,i1+1) = 0; }
      }

    fAmp_max_ += variation_func_max ;
    fTim_max_ += variation_tim_max ;
    fPed_max_ += variation_ped_max ;
    
    //! Then we loop on samples to be fitted
    for( int i = num_fit_min ; i <= num_fit_max ; i++) {
      //if(i>fsamp_edge_fit && i<num_fit_min) continue ; // remove front edge samples
      //! calculate function to be fitted
      func = pulseShapeFunction( (double)i  ) ;
      //! then calculate derivatives of function to be fitted
      double dt =(double)i - fTim_max_ ;
      if(dt > -alfabeta_)  {      
	double dt_sur_beta = dt/fBeta_ ;
	double variable = (double)1. + dt/alfabeta_ ;
	double expo = exp(-dt_sur_beta) ;	 
	double puissance = pow(variable,fAlpha_) ;
	
	db[0]=un_sur_sigma*puissance*expo ;
	db[1]=fAmp_max_*db[0]*dt_sur_beta/(alfabeta_*variable) ;
      }
      else {
	db[0]=0. ; db[1]=0. ; 
      }
      db[2]=un_sur_sigma ;
      //! compute matrix elements DM1
      for(int i1=0 ; i1<3 ; i1++) {
	for(int j1=i1 ; j1<3 ; j1++) { 
	  //double & fast(int row, int col);
	  DM1_.fast(j1+1,i1+1) += db[i1]*db[j1]; }
      }
      //! compute delta
      delta = (samples[i]-func)*un_sur_sigma ;
      //! compute vector elements PROD
      for(int ii=0 ; ii<3 ;ii++) {temp_[ii] += delta*db[ii] ;}
      chi2 += delta * delta ;
    }//! end of loop on samples 
    
    int fail=0 ;
    DM1_.invert(fail) ;
      if(fail != 0.) {
      //just a guess from the value of the parameters in the previous interaction;
      //printf("wH4PulseFitWithFunction =====> determinant error --> No Fit Provided !\n") ;
      return -101 ;
    }
/*     for(int i1=0 ; i1<3 ; i1++) { */
/*       for(int j1=0 ; j1<3 ; j1++) {  */
/* 	//double & fast(int row, int col); */
/* 	std::cout<<"inverted: "<<DM1[j1][i1]<<std::endl;;} */
/*     } */
/*     std::cout<<"vector temp: "<< temp[0]<<" "<<temp[1]<<" "<<temp[2]<<std::endl; */
    //! compute variations of parameters fAmp_max and fTim_max 
    HepVector PROD = DM1_*temp_ ;
    //    std::cout<<"vector PROD: "<< PROD[0]<<" "<<PROD[1]<<" "<<PROD[2]<<std::endl;
    variation_func_max = PROD[0] ;
    variation_tim_max = PROD[1] ;
    variation_ped_max = PROD[2] ;
    //chi2 = chi2/((double)nsamp_used - 3.) ;
  }//!end of loop on iterations       
  
  //!      results of the fit are calculated

  //!   protection again diverging fit 
  if( variation_func_max > 2000. || variation_func_max < -1000. ) {
    //! use the first guess
    InitFitParameters(samples,max_sample);
    return -102 ;
  }
   
  fAmp_max_ += variation_func_max ;
  fTim_max_ += variation_tim_max ;
  fPed_max_ += variation_ped_max ;

  //std::cout <<"chi2: "<<chi2<<" ampl: "<<fAmp_max_<<" time: "<<fTim_max_<<" pede: "<<fPed_max_<<std::endl;
  return chi2;
}

template<class C> void EcalUncalibRecHitFixedAlphaBetaAlgo<C>::SetAlphaBeta( double alpha, double beta){
  fAlpha_ = alpha;
  fBeta_=  beta;
  alfabeta_ = fAlpha_*fBeta_;
}

#endif
