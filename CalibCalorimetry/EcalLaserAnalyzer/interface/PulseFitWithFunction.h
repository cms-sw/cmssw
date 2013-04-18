//-----------------------------------------------------------------------
//----------------------------------------------------------------------
// File PulseFitWithFunction.h
// ===========================================================
// ==                                                       ==
// ==     Class for a LIGHT weights method                  ==
// ==                                                       ==
// ==  Date:   July 16th 2003                               ==
// ==  Author: Patrick Jarry                                ==
// ==            ==
// ==                                                       ==
// ===========================================================
/* This routine is used to fit the signal line
      shape of CMS barrel calorimeter
  The method used is the one described in note LPC 84-30 (Billoir 1984) :
    "Methode d'ajustement dans un probleme a parametrisation hierarchisee"
  In this class we calculate the amplitude maximum and the time of arrival
  of this maximum (done with function fit_electronic)
 */

#ifndef PulseFitWithFunction_H
#define PulseFitWithFunction_H

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/PulseFit.h>

class PulseFitWithFunction: public TObject 
{
 public:
  // Default Constructor, mainly for Root
  PulseFitWithFunction() ;

  // Destructor: Does nothing
  virtual ~PulseFitWithFunction() ;

  // Initialize 
  virtual void init(int,int,int,int,double,double) ;

  // Compute amplitude of a channel

  virtual double doFit(double *) ;
  
  double fFunc_max ; // amplitude maximum as input of fit
  double fTim_max ; // time of amplitude maximum as input of fit
  double fAmp_fitted_max ; // amplitude maximum fitted
  double fTim_fitted_max ; // time of amplitude maximum fitted 
  double fValue_tim_max ; // value of time of arrival of maximum from pol3 fit
  int    fNumber_samp_max ; // number of the sample which is maximum 
  double fSigma_ped ; // sigma of pedestal to be used in fit

  double getAmpl_parab() { return amp_parab; }
  double getTime_parab() { return tim_parab; }

  double getAmpl() { return fAmp_fitted_max; }
  double getTime() { return fTim_fitted_max; }

  double getMax_parab() { return amp_max; }
  int getSampMax_parab() { return imax; }

 private:	

  double amp_max , amp_parab , tim_parab;
  int imax;
  
  int fNsamples ; // maximum number of samples into framelegth

  double  fAlpha_laser ;
  double  fBeta_laser ;
  double  fAlpha_beam ;
  double  fBeta_beam ;
  double  fAlpha ;
  double  fBeta ;
  int     fNb_iter ; // maximum number of iterations
  int     fNum_samp_bef_max  ; // number of samples before maximum sample
  int     fNum_samp_after_max  ; // number of samples after  maximum sample
 
  double Fit_electronic(int, double *,double ) ;
  void Fit_parab(double *,int,int,double * ) ;
  double Electronic_shape(double) ;
  
  

  //  ClassDef(PulseFitWithFunction,1)     //!< The processed part of the class is persistant
} ;

#endif



//-----------------------------------------------------------------------
//----------------------------------------------------------------------
