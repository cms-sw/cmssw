// Class:      SiStripPulseShape
//
/**\class SiStripPulseShape SiStripPulseShape.h myTestArea/SiStripPulseShape/src/SiStripPulseShape.h

 Description: analog pulse shape at the ouput of the APV. 

 Implementation:
     This class allows to access the pulse shape at the APV. This is  usefull to take into account
     timing effects in the tracker.
*/
//
// Original Author:  Christophe Delaere
//         Created:  Thu Nov  5 17:02:15 CEST 2006
// $Id: SiStripPulseShape.h,v 1.2 2006/12/14 16:49:19 delaer Exp $
//
//

#ifndef SiStripPulseShape_h_
#define SiStripPulseShape_h_

#include <TF1.h>

double fpeak(double *x, double *par)
    {
      if(x[0]+par[1]<0) return par[0];
      return par[0]+par[2]*(x[0]+par[1])*TMath::Exp(-(x[0]+par[1])/par[3]);
    }

double fdeconv(double *x, double *par)
    {
      double xm = par[4]*(x[0]-25);
      double xp = par[4]*(x[0]+25);
      double xz = par[4]*x[0];
      return 1.2131*fpeak(&xp,par)-1.4715*fpeak(&xz,par)+0.4463*fpeak(&xm,par);
    }

double fpeak_convoluted(double *x, double *par)
    {
     TF1 f("peak_convoluted",fpeak,0,200,4);
     return f.Integral(x[0]-par[4]/2.,x[0]+par[4]/2.,par,1.)/(par[4]);
    } 

double fdeconv_convoluted(double *x, double *par)
    {
      double xm = (x[0]-25);
      double xp = (x[0]+25);
      double xz = x[0];
      return 1.2131*fpeak_convoluted(&xp,par)-1.4715*fpeak_convoluted(&xz,par)+0.4463*fpeak_convoluted(&xm,par);
    }

class SiStripPulseShape
{
  public: 
    enum mode {peak,deconvolution};
    SiStripPulseShape():mode_(deconvolution) {}
    virtual ~SiStripPulseShape() {}
    inline void setMode(const mode theMode) { mode_=theMode; }
    inline mode getMode() const { return mode_; } 
    inline double getNormalizedValue(const double& t) const
    {
      double parameters[5]={0.,-2.82,0.066,50,20};
      double time = t;
      switch(mode_) {
       case peak:
        {
	  return fpeak_convoluted(&time,parameters);
	}
       case deconvolution:
        {
	  return fdeconv_convoluted(&time,parameters);
	}
      }
    }

  private:
    mode mode_;
};

#endif
