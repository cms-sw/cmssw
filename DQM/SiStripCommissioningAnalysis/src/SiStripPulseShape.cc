#include "DQM/SiStripCommissioningAnalysis/interface/SiStripPulseShape.h"
#include <TF1.h>
#include <TMath.h>

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
     return f.IntegralError(x[0]-par[4]/2.,x[0]+par[4]/2.,par,0,1.)/(par[4]);
    } 

double fdeconv_convoluted(double *x, double *par)
    {
      double xm = (x[0]-25);
      double xp = (x[0]+25);
      double xz = x[0];
      return 1.2131*fpeak_convoluted(&xp,par)-1.4715*fpeak_convoluted(&xz,par)+0.4463*fpeak_convoluted(&xm,par);
    }

