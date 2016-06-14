#ifndef _positionfit_
#define _positionfit_

#include "TMath.h"
#include "TMinuit.h"
#include <vector>
#include <iostream>

struct HGCPositionFitResult_t
{
  Int_t status;
  Double_t xff,      xffErr;
  Double_t yff,      yffErr;
  Double_t zv,       zvErr;
  Double_t theta,    thetaErr;
  Double_t chi2,prob;
  Int_t ndf;
};


enum _hgcpos_fitparams { PAR_XFF, PAR_YFF, PAR_ZV };
Int_t   _hgcpos_npts;
Float_t _hgcpos_zff;
Float_t _hgcpos_x[100], _hgcpos_y[100], _hgcpos_z[100];
Float_t _hgcerr_x[100], _hgcerr_y[100];

//
void _hgcposfit_chiSquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  //calculate chisquare
  Double_t chisq(0);
  for (int i=0;i<_hgcpos_npts; i++) 
    {
      Double_t xExp=par[PAR_XFF]*(_hgcpos_z[i]-par[PAR_ZV])/(_hgcpos_zff-par[PAR_ZV]);
      chisq += TMath::Power((_hgcpos_x[i]-xExp)/_hgcerr_x[i],2);
      
      Double_t yExp=par[PAR_YFF]*(_hgcpos_z[i]-par[PAR_ZV])/(_hgcpos_zff-par[PAR_ZV]);
      chisq += TMath::Power((_hgcpos_y[i]-yExp)/_hgcerr_x[i],2);
    }
  
  f=chisq;
  return;
}

//
TMinuit *_hgcposfit_init(Double_t zffVal=317.0,int verbose=-1)
{
  _hgcpos_zff=zffVal;

  //init minuit class
  TMinuit *minuit = new TMinuit(3);
  minuit->SetPrintLevel(verbose);
  minuit->SetFCN(_hgcposfit_chiSquare);
  minuit->DefineParameter(PAR_XFF, "xff", _hgcpos_x[0], 0.01, -500, 500);
  minuit->DefineParameter(PAR_YFF, "yff", _hgcpos_y[0], 0.01, -500, 500);
  minuit->DefineParameter(PAR_ZV,  "zv",  0.,           0.01, -500, 500);

  return minuit;
} 

//
HGCPositionFitResult_t _hgcposfit_run(TMinuit *minuit,int verbose=-1)
{
  HGCPositionFitResult_t result;
  if(_hgcpos_npts<2)
    {
      result.status=-1;
      std::cout << "_hgcposfit_run : at least two points needed to fit a straight line..." << std::endl;
      return result;
    }

  //dump points used for fit, if required
  if(verbose>0)
    {
      std::cout << "_hgcposfit_run" << std::endl;
      for(Int_t i=0; i<_hgcpos_npts; i++)
	std::cout << "    " << _hgcpos_x[i] << " " << _hgcpos_y[i] << " " << _hgcpos_z[i] << std::endl;
    }

  //minimize the chi2
  result.status=minuit->Migrad();

  //save result if mininimization was ok
  if(result.status==0)
    {
      minuit->GetParameter(PAR_XFF, result.xff, result.xffErr);
      minuit->GetParameter(PAR_YFF, result.yff, result.yffErr);
      minuit->GetParameter(PAR_ZV,  result.zv,  result.zvErr);
      
      Double_t rhoff=TMath::Sqrt(result.xff*result.xff+result.yff*result.yff);
      result.theta=TMath::ATan2(rhoff,_hgcpos_zff+result.zv);

      Int_t npar(3);
      Double_t par[]={result.xff,result.yff,result.zv};
      _hgcposfit_chiSquare(npar,0,result.chi2,par,0);
      result.ndf=_hgcpos_npts-3;
      result.prob=TMath::Prob(result.chi2,result.ndf);
    }
  
  //all done
  return result;
}



#endif
