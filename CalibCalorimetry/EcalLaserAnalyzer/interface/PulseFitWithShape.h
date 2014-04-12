//-----------------------------------------------------------------------
//----------------------------------------------------------------------
// File PulseFitWithShape.h

#ifndef PulseFitWithShape_H
#define PulseFitWithShape_H
#include "TObject.h"
#include<vector>

class PulseFitWithShape: public TObject 
{
 public:
  // Default Constructor, mainly for Root
  PulseFitWithShape() ;

  // Destructor: Does nothing
  virtual ~PulseFitWithShape() ;

  // Initialize 
  virtual void init(int,int,int,int,int,const std::vector <double>&,double) ;

  // Compute amplitude of a channel

  virtual double doFit(double *,double *cova=0) ;
  
  double fAmp_fitted_max ; // amplitude maximum fitted
  double fTim_fitted_max ; // time of amplitude maximum fitted 
 
  double getAmpl() { return fAmp_fitted_max; }
  double getTime() { return fTim_fitted_max; }


 private:	

  int fNsamples ;  // maximum number of samples into framelegth
  int fNsamplesShape ;  // maximum number of samples into framelegth
  double fNoise;
  
  std::vector < double > pshape;  
  std::vector < double > dshape; 

  int     fNb_iter ; // maximum number of iterations
  int     fNum_samp_bef_max  ; // number of samples before maximum sample
  int     fNum_samp_after_max  ; // number of samples after  maximum sample
 
  
  // ClassDef(PulseFitWithShape,1)     //!< The processed part of the class is persistant

} ;

#endif



//-----------------------------------------------------------------------
//----------------------------------------------------------------------
