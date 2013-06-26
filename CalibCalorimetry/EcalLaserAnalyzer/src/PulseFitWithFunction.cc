/* 
 *  \class PulseFitWithFunction
 *
 *  $Date: 2012/02/09 10:08:09 $
 *  \author: Patrick Jarry - CEA/Saclay
 */


// File PulseFitWithFunction.cxx
// ===========================================================
// ==                                                       ==
// ==     Class for a function fit method                   ==
// ==                                                       ==
// ==  Date:   July 17th 2003                               ==
// ==  Author: Patrick Jarry                                ==
// ==                                                       ==
// ==                                                       ==
// ===========================================================

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/PulseFitWithFunction.h>

#include <iostream>
#include "TMath.h"

//ClassImp(PulseFitWithFunction)


// Constructor...
PulseFitWithFunction::PulseFitWithFunction()
{ 
 
  fNsamples               =  0;
  fNum_samp_bef_max       =  0;
  fNum_samp_after_max     =  0;
}

// Destructor
PulseFitWithFunction::~PulseFitWithFunction()
{ 
}

// Initialisation

void PulseFitWithFunction::init(int n_samples,int samplb,int sampla,int niter,double alfa,double beta)
{
  //printf("\n");
  //printf(" =========================================================\n");
  //printf(" ==     Initialising the function fit method            ==\n");
  //printf(" ==          PulseFitWithFunction::init                 ==\n");
  //printf(" ==                                                     ==\n");

  
  fNsamples   = n_samples ;
  fAlpha_laser = alfa;
  fBeta_laser = beta ;
  fAlpha_beam = 0.98 ;
  fBeta_beam = 2.04 ;
  fNb_iter = niter ;
  fNum_samp_bef_max = samplb ;
  fNum_samp_after_max = sampla  ;
  //printf(" ==  # samples used = %3d                               ==\n",fNsamples);
  //printf(" ==  #sample before max= %1d and #sample after maximum= %1d ==\n",fNum_samp_bef_max,fNum_samp_after_max);
  //printf(" ==           alpha= %5.4f       beta= %5.4f          ==\n",fAlpha_laser,fBeta_laser);
  //printf(" =========================================================\n\n");
  return ;
 }

// Compute the amplitude using as input the Crystaldata  
double PulseFitWithFunction::doFit(double *adc)
{
  double parout[4]; // amp_max ;
    //double amp_parab , tim_parab ;
  double chi2;
  //int imax ;
//
// first  one has to get starting point first with parabolic fun// //  
  Fit_parab(&adc[0],3,fNsamples,parout) ;
  amp_parab = parout[0] ;
  tim_parab = parout[1] ;
  imax = (int)parout[2] ;
  amp_max = parout[3] ;
  //printf("amp_parab= %f tim_parab=%f amp_max=%f imax=%d\n",amp_parab,tim_parab,amp_max,imax); 
  fNumber_samp_max=imax ;
//
  
  if(amp_parab < 1.) {
     tim_parab = (double)imax ;
      amp_parab = amp_max ;
  }
//
  fValue_tim_max= tim_parab ;
  fFunc_max= amp_parab ;
  fTim_max=tim_parab ;

//  here to fit maximum amplitude and time of arrival ...

  chi2 = Fit_electronic(0,&adc[0],8.) ;
                                   // adc is an array to be filled with samples
                                   // 0 is for Laser (1 for electron)  
                                   // 8 is for sigma of pedestals 
                                   // which (can be computed)


  //  double amplitude = fAmp_fitted_max ; // amplitude fitted 
  // double time = fTim_fitted_max ; // time fitted

 

  return chi2 ; // return amplitude fitted

}

//-----------------------------------------------------------------------
//----------------------------------------------------------------------

/*************************************************/
double PulseFitWithFunction::Fit_electronic(int data , double* adc_to_fit , double sigmas_sample) {
  // fit electronic function from simulation
  // parameters fAlpha and fBeta are fixed and fit is providing
  // the maximum amplitude ( fAmp_fitted_max ) and the time of
  // the maximum amplitude ( fTim_fitted_max) 
  // initialization of parameters
  double chi2=0;
  double d_alpha, d_beta ;
  // first initialize parameters fAlpha and fBeta ( depending of beam or laser)
  fAlpha = fAlpha_laser ;
  fBeta = fBeta_laser ;
  if(data == 1) { 
    fAlpha = fAlpha_beam ;
    fBeta = fBeta_beam ;
  }
  //
  fAmp_fitted_max = 0. ;
  fTim_fitted_max = 0. ;
  double un_sur_sigma = 1./fSigma_ped ;
  double variation_func_max = 0. ;
  double variation_tim_max = 0. ;
  //
  if(fValue_tim_max > 20. || fValue_tim_max  < 3.) {
	  fValue_tim_max = fNumber_samp_max ;
  }
  int num_fit_min =(int)(fValue_tim_max - fNum_samp_bef_max) ;
  int num_fit_max =(int)(fValue_tim_max + fNum_samp_after_max) ;
  //

  if( sigmas_sample > 0. ) un_sur_sigma = 1./sigmas_sample;
  else un_sur_sigma = 1.;

  double func,delta ;
  //          Loop on iterations        
  for (int iter=0 ; iter < fNb_iter ; iter ++) {
   
    //          initialization inside iteration loop !
    chi2 = 0. ;
    double d11 = 0. ;
    double d12 = 0. ;
    double d22 = 0. ;
    double z1 = 0. ;
    double z2 = 0. ;
    fFunc_max += variation_func_max ;
    fTim_max += variation_tim_max ;
    int nsamp_used = 0 ;
    //
   
    // Then we loop on samples to be fitted


    for( int i = num_fit_min ; i < num_fit_max+1 ; i++) {
      // calculate function to be fitted
     
      func = Electronic_shape( (double)i  ) ;
      
      // then calculate derivatives of function to be fitted
      double dt =(double)i - fTim_max ;
      double alpha_beta = fAlpha*fBeta  ;
     
     if(dt > -alpha_beta)  {      
      double dt_sur_beta = dt/fBeta ;
	 
      double variable = (double)1. + dt/alpha_beta ;
	  double expo = TMath::Exp(-dt_sur_beta) ;	 
	 
	  double puissance = TMath::Power(variable,fAlpha) ;
      d_alpha=un_sur_sigma*puissance*expo ;
      d_beta=fFunc_max*d_alpha*dt_sur_beta/(alpha_beta*variable) ;
     }
      else {
	continue ;
     }
     
      nsamp_used ++ ; // number of samples used inside the fit
      // compute matrix elements D (symetric --> d12 = d21 )
      d11 += d_alpha*d_alpha ;
      d12 += d_alpha*d_beta ;
      d22 += d_beta*d_beta ;
      // compute delta
      delta = (adc_to_fit[i]-func)*un_sur_sigma ;
      // compute vector elements Z
      z1 += delta*d_alpha ;
      z2 += delta*d_beta ;
      chi2 += delta * delta ;
    } //             end of loop on samples  
    double denom = d11*d22-d12*d12 ;
    if(denom == 0.) {
      //printf( "attention denom = 0 signal pas fitte \n") ;
      return 101 ;
    }
    if(nsamp_used < 3) {
      //printf( "Attention nsamp = %d ---> no function fit provided \n",nsamp_used) ;
      return 102 ;
    }
    // compute variations of parameters fAmp_max and fTim_max 
    variation_func_max = (z1*d22-z2*d12)/denom ;
    variation_tim_max = (-z1*d12+z2*d11)/denom ;
    chi2 = chi2/((double)nsamp_used - 2.) ;
  } //      end of loop on iterations       
  // results of the fit are calculated 
  fAmp_fitted_max = fFunc_max + variation_func_max ;
  fTim_fitted_max = fTim_max + variation_tim_max ;
  //
  return chi2 ;
}

//-----------------------------------------------------------------------
//----------------------------------------------------------------------

double  PulseFitWithFunction::Electronic_shape(double tim)
{
  // electronic function (from simulation) to fit ECAL pulse shape
  double func_electronic,dtsbeta,variable,puiss;
  double albet = fAlpha*fBeta ;
  if( albet <= 0 ) return( (Double_t)0. );
  double dt = tim-fTim_max ;
  if(dt > -albet)  {
    dtsbeta=dt/fBeta ;
    variable=1.+dt/albet ;
	puiss=TMath::Power(variable,fAlpha);
	func_electronic=fFunc_max*puiss*TMath::Exp(-dtsbeta);
  }
  else func_electronic = 0. ;
  //
  return func_electronic ;
}
void PulseFitWithFunction::Fit_parab(Double_t *ampl,Int_t nmin,Int_t nmax,Double_t *parout)
{
/* Now we calculate the parabolic adjustement in order to get        */
/*    maximum and time max                                           */
 
  double denom,dt,amp1,amp2,amp3 ;
  double ampmax = 0. ;				
  int imax = 0 ;
  int k ;
/*
                                                                   */	  
  for ( k = nmin ; k < nmax ; k++) {
    //printf("ampl[%d]=%f\n",k,ampl[k]);
    if (ampl[k] > ampmax ) {
      ampmax = ampl[k] ;
      imax = k ;
    }
  }
	amp1 = ampl[imax-1] ;
	amp2 = ampl[imax] ;
	amp3 = ampl[imax+1] ;
	denom=2.*amp2-amp1-amp3  ;
/* 	       					             */	      
	if(denom>0.){
	  dt =0.5*(amp3-amp1)/denom  ;
	}
	else {
	  //printf("denom =%f\n",denom)  ;
	  dt=0.5  ;
	}
/* 						                     */	       
/* ampmax correspond au maximum d'amplitude parabolique et dt        */
/* decalage en temps par rapport au sample maximum soit k + dt       */
		
	parout[0] =amp2+(amp3-amp1)*dt*0.25 ;
	parout[1] = (double)imax + dt ;
        parout[2] = (double)imax ;
	parout[3] = ampmax ;
return ;
}
