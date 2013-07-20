/* 
 *  \class PulseFitWithShape
 *
 *  $Date: 2013/04/19 22:19:23 $
 *  \author: Julie Malcles - CEA/Saclay
 */


// File PulseFitWithShape.cxx

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/PulseFitWithShape.h>

#include <iostream>
#include "TMath.h"
#include <cmath>

//ClassImp(PulseFitWithShape)


// Constructor...
PulseFitWithShape::PulseFitWithShape()
{ 
 
  fNsamples               =  0;
  fNsamplesShape          =  0;
  fNum_samp_bef_max       =  0;
  fNum_samp_after_max     =  0;
  fNoise                  = 0.0;
}

// Destructor
PulseFitWithShape::~PulseFitWithShape()
{ 
}

// Initialisation

void PulseFitWithShape::init(int n_samples,int samplb,int sampla,int niter,int n_samplesShape, const std::vector<double>& shape, double nois)
{
 
  fNsamples   = n_samples ;
  fNsamplesShape   = n_samplesShape ;
  fNb_iter = niter ;
  fNum_samp_bef_max = samplb ;
  fNum_samp_after_max = sampla  ;


  if( fNsamplesShape < fNum_samp_bef_max+fNum_samp_after_max+1){
    std::cout<<"PulseFitWithShape::init: Error! Configuration samples in fit greater than total number of samples!" << std::endl;
  }

  for(int i=0;i<fNsamplesShape;i++){
    pshape.push_back(shape[i]);
    dshape.push_back(0.0);
  }

  fNoise=nois;
  return ;
 }

// Compute the amplitude using as input the Crystaldata  

double PulseFitWithShape::doFit(double *adc, double *cova)
{

  // xpar = fit paramaters
  //     [0] = signal amplitude
  //     [1] = residual pedestal
  //     [2] = clock phase

  bool useCova=true;
  if(cova==0) useCova=false;

  double xpar[3]; 
  double chi2;

  fAmp_fitted_max = 0. ;
  fTim_fitted_max = 0. ;
  
  // for now don't fit pedestal

  xpar[1]=0.0;
  
  // Sample noise. If the cova matrix is defined, use it :

  double noise=fNoise;
  //if(cova[0] > 0.) noise=1./sqrt(cova[0]);
  
  xpar[0]=0.;
  xpar[2]=0.;


  // first locate max:

  int imax=0;
  double amax=0.0;
  for(int i=0; i<fNsamples; i++){
    if (adc[i]>amax){
      amax=adc[i];
      imax=i;
    }
  }
  
  // Shift pulse shape and calculate its derivative:
  
  double qms=0.;
  int ims=0;
  
  // 1) search for maximum
  
  for(int is=0; is<fNsamplesShape; is++){
    if(pshape[is] > qms){
      qms=pshape[is];
      ims=is;
    }

  // 2) compute shape derivative :
    
    if(is < fNsamplesShape-2)
      dshape[is]= (pshape[is+2]-pshape[is])*12.5;
    else
      dshape[is]=dshape[is-1]; 
  }

  // 3) compute pol2 max

  double sq1=pshape[ims-1];
  double sq2=pshape[ims];
  double sq3=pshape[ims+1];
  
  double a2=(sq3+sq1)/2.0-sq2;
  double a1=sq2-sq1+a2*(1-2*ims);
 
  
  double t_ims=0;
  if(a2!=0) t_ims=-a1/(2.0*a2);


  // Starting point of the fit : qmax and tmax given by a
  // P2 fit on 3 max samples.
  
  double qm=0.;
  int im=0;
 
  int nsamplef=fNum_samp_bef_max + fNum_samp_after_max +1 ; // number of samples used in the fit
  int nsampleo=imax-fNum_samp_bef_max;  // first sample number = sample max-fNum_samp_bef_max 
  
  for(int is=0; is<nsamplef; is++){

    if(adc[is+nsampleo] > qm){
      qm=adc[is+nsampleo];
      im=nsampleo+is;
    }
  }

  double tm;
  if(qm > 5.*noise){
    if(im >= nsamplef+nsampleo) im=nsampleo+nsamplef-1;
    double q1=adc[im-1];
    double q2=adc[im];
    double q3=adc[im+1];
    double y2=(q1+q3)/2.-q2;
    double y1=q2-q1+y2*(1-2*im);
    double y0=q2-y1*(double)im-y2*(double)(im*im);
    tm=-y1/2./y2;
    xpar[0]=y0+y1*tm+y2*tm*tm;
    xpar[2]=(double)ims/25.-tm;
  }

  double chi2old=999999.;
  chi2=99999.;
  int nsfit=nsamplef;
  int iloop=0;
  int nloop=fNb_iter;
  if(qm <= 5*noise)nloop=1; // Just one iteration for very low signal

  double *resi;
  resi= new double[fNsamples];  
  for (int j=0;j<fNsamples;j++) resi[j]=0;

  while(std::fabs(chi2old-chi2) > 0.1 && iloop < nloop)
    {
      iloop++;
      chi2old=chi2;
      
      double c=0.;
      double y1=0.;
      double s1=0.;
      double s2=0.;
      double ys1=0.;
      double sp1=0.;
      double sp2=0.;
      double ssp=0.;
      double ysp=0.;
      
      for(int is=0; is<nsfit; is++)
	{
	  int iis=is;
	  iis=is+nsampleo;
	  
	  double t1=(double)iis+xpar[2];
	  double xbin=t1*25.;
	  int ibin1=(int)xbin;
	  
	  if(ibin1 < 0) ibin1=0;

	  double amp1,amp11,amp12,der1,der11,der12;

	  if(ibin1 <= fNsamplesShape-2){     // Interpolate shape values to get the right number :
	    
	    int ibin2=ibin1+1;
	    double xfrac=xbin-ibin1;
	    amp11=pshape[ibin1];
	    amp12=pshape[ibin2];
	    amp1=(1.-xfrac)*amp11+xfrac*amp12;
	    der11=dshape[ibin1];
	    der12=dshape[ibin2];
	    der1=(1.-xfrac)*der11+xfrac*der12;
	    
	  }else{                            // Or extraoplate if we are outside the array :
	    
	    amp1=pshape[fNsamplesShape-1]+dshape[fNsamplesShape-1]*
	      (xbin-double(fNsamplesShape-1))/25.;
	    der1=dshape[fNsamplesShape-1];
	  }
	  
	  if( useCova ){     // Use covariance matrix: 
	    for(int js=0; js<nsfit; js++)
	      {
		int jjs=js;
		jjs+=nsampleo;
		
		t1=(double)jjs+xpar[2];
		xbin=t1*25.;
		ibin1=(int)xbin;
		if(ibin1 < 0) ibin1=0;
		if(ibin1 > fNsamplesShape-2)ibin1=fNsamplesShape-2;
		int ibin2=ibin1+1;
		double xfrac=xbin-ibin1;
		amp11=pshape[ibin1];
		amp12=pshape[ibin2];
		double amp2=(1.-xfrac)*amp11+xfrac*amp12;
		der11=dshape[ibin1];
		der12=dshape[ibin2];
		double der2=(1.-xfrac)*der11+xfrac*der12;
		c=c+cova[js*fNsamples+is];
		y1=y1+adc[iis]*cova[js*fNsamples+is];
		s1=s1+amp1*cova[js*fNsamples+is];
		s2=s2+amp1*amp2*cova[js*fNsamples+is];
		ys1=ys1+adc[iis]*amp2*cova[js*fNsamples+is];
		sp1=sp1+der1*cova[is*fNsamples+js];
		sp2=sp2+der1*der2*cova[js*fNsamples+is];
		ssp=ssp+amp1*der2*cova[js*fNsamples+is];
		ysp=ysp+adc[iis]*der2*cova[js*fNsamples+is];
	      }
	  }else { // Don't use covariance matrix: 
	    c++;
	    y1=y1+adc[iis];
	    s1=s1+amp1;
	    s2=s2+amp1*amp1;
	    ys1=ys1+amp1*adc[iis];
	    sp1=sp1+der1;
	    sp2=sp2+der1*der1;
	    ssp=ssp+der1*amp1;
	    ysp=ysp+der1*adc[iis];
	  }
	}
      xpar[0]=(ysp*ssp-ys1*sp2)/(ssp*ssp-s2*sp2);
      xpar[2]+=(ysp/xpar[0]/sp2-ssp/sp2);
   
      for(int is=0; is<nsfit; is++){
	int iis=is;
	iis=is+nsampleo;
	
	double t1=(double)iis+xpar[2];
	double xbin=t1*25.;
	int ibin1=(int)xbin;
	if(ibin1 < 0) ibin1=0;
	
	if(ibin1 < 0) ibin1=0;
	if(ibin1 > fNsamplesShape-2)ibin1=fNsamplesShape-2;
	int ibin2=ibin1+1;
	double xfrac=xbin-ibin1;
	double amp11=xpar[0]*pshape[ibin1];
	double amp12=xpar[0]*pshape[ibin2];
	double amp1=xpar[1]+(1.-xfrac)*amp11+xfrac*amp12;
	resi[iis]=adc[iis]-amp1;
      }

      chi2=0.;
      for(int is=0; is<nsfit; is++){
	int iis=is;
	iis+=nsampleo;
	
	if( useCova ){
	  for(int js=0; js<nsfit; js++){
	    int jjs=js;
	    jjs+=nsampleo;
	    chi2+=resi[iis]*resi[jjs]*cova[js*fNsamples+is];
	  }

	}else chi2+=resi[iis]*resi[iis];
      }
    }
  
  fAmp_fitted_max = xpar[0];
  fTim_fitted_max = (double)t_ims/25.-xpar[2];
  
  return chi2 ;

}

