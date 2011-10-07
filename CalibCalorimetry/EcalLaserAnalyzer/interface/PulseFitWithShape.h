//-----------------------------------------------------------------------
//----------------------------------------------------------------------
// File PulseFitWithShape.h

#ifndef PulseFitWithShape_H
#define PulseFitWithShape_H
#include<vector>
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/PulseFits.h"
using namespace std;

class PulseFitWithShape: public PulseFits
{
 public:
  // Default Constructor, mainly for Root
  PulseFitWithShape() ;

  // Destructor: Does nothing
  virtual ~PulseFitWithShape() ;

  // Initialize 
  virtual void init(int,int,int,int,int,int,vector <double>,double) ;

  // Compute amplitude of a channel

  virtual double doFit(double *,double *cova) ;
  virtual double doFit(double *) ;

  double fAmp_fitted_max ; // amplitude maximum fitted
  double fTim_fitted_max ; // time of amplitude maximum fitted 
 
  double getAmpl() { return fAmp_fitted_max; }
  double getTime() { return fTim_fitted_max; }
  void setFitPed(bool);


 private:	

  int fNsamples ;  // maximum number of samples into framelegth
  int fNsamplesShape ;  // maximum number of samples into framelegth
  double fNoise;
  bool fFitPed;
  bool debug;

  vector < double > pshape;  
  vector < double > dshape; 

  int     fNb_iter ; // maximum number of iterations
  int     fNum_samp_bef_max  ; // number of samples before maximum sample
  int     fNum_samp_after_max  ; // number of samples after  maximum sample
  int     fNum_presample;
  
  //ClassDef(PulseFitWithShape,1)     //!< The processed part of the class is persistant

} ;

#endif



//-----------------------------------------------------------------------
//----------------------------------------------------------------------
