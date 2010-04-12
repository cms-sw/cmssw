/* 
 *  \class TPNFit
 *
 *  $Date: 2009/06/02 12:55:21 $
 *  \author: Patrice Verrecchia - CEA/Saclay
 */

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TPNFit.h>

#include <iostream>
#include "TVectorD.h"
#include "TF1.h"
#include "TH1D.h"
#include "TMath.h"
#include "TFile.h"

//ClassImp(TPNFit)


// Constructor...
TPNFit::TPNFit()
{ 
 
  fNsamples               =  0;
  fNum_samp_bef_max       =  0;
  fNum_samp_after_max     =  0;
}

// Destructor
TPNFit::~TPNFit()
{ 
}

void TPNFit::init(int nsamples, int firstsample, int lastsample)
{
  fNsamples   = nsamples ;
  fNum_samp_bef_max = firstsample ;
  fNum_samp_after_max = lastsample  ;
  //printf("nsamples=%d firstsample=%d lastsample=%d\n",nsamples,firstsample,lastsample);

  if(fNsamples > NMAXSAMPPN)
          printf("number of PN samples exceed maximum\n");

  for(int k=0;k<NMAXSAMPPN;k++)
       t[k]= (double) k;

  htmp = new TH1D("htmp","htmp",NMAXSAMPPN,0.,double(NMAXSAMPPN));
  fPN = new TF1("fPN",fitPN_tp,0.,double(NMAXSAMPPN),5);

  return ;
}

double TPNFit::doFit(int maxsample, double *adc)
{
  double chi2=0.;
  ampl=0.; timeatmax=0.;

  //printf("maxsample=%d\n",maxsample);
  firstsample= maxsample - fNum_samp_bef_max;
  lastsample= maxsample +  fNum_samp_after_max;

  if(firstsample <= 0) return 101;
  if(lastsample >= fNsamples) lastsample=fNsamples-1;
  if(lastsample-firstsample < 1) return 102;
  int nval= lastsample-firstsample +1;
  //printf("firstsample=%d lastsample=%d nval=%d\n",
  //                        firstsample,lastsample,nval);
  int testneg=0;
  for(int kn=firstsample;kn<=lastsample;kn++) {
    //printf("adc[%d]=%f\n",kn,adc[kn]);
	  if(adc[kn] < 0.) testneg=1;
  }
  if(testneg == 1) return 103;

  for(int i=firstsample;i<=lastsample;i++) {
     val[i-firstsample]= adc[i];
     fv1[i-firstsample]= 1.;
     fv2[i-firstsample]= (double) (i);
     fv3[i-firstsample]= ((double)(i))*((double)(i));
  }

  TVectorD y(nval,val);
  //y.Print();
  TVectorD f1(nval,fv1);
  //f1.Print();
  TVectorD f2(nval,fv2);
  //f2.Print();
  TVectorD f3(nval,fv3);
  //f3.Print();
 
  double bj[3];
  bj[0]= f1*y; bj[1]=f2*y; bj[2]= f3*y;
  TVectorD b(3,bj);
  //b.Print();

  double aij[9];
  aij[0]= f1*f1; aij[1]=f1*f2; aij[2]=f1*f3;
  aij[3]= f2*f1; aij[4]=f2*f2; aij[5]=f2*f3;
  aij[6]= f3*f1; aij[7]=f3*f2; aij[8]=f3*f3;
  TMatrixD a(3,3,aij);
  //a.Print();
 
  double det1;
  a.InvertFast(&det1);
  //a.Print();

  TVectorD c= a*b;
  //c.Print();

  double *par= c.GetMatrixArray();
  //printf("par0=%f par1=%f par2=%f\n",par[0],par[1],par[2]);
  if(par[2] != 0.) {
    timeatmax= -par[1]/(2.*par[2]);
    ampl= par[0]-par[2]*(timeatmax*timeatmax);
  }
  //printf("amp=%f time=%f\n",amp_max,timeatmax);
     
  if(ampl <= 0.) {
      ampl=adc[maxsample];
      return 1.;
  }

  if((int)timeatmax > lastsample)
      return 103;
  if((int)timeatmax < firstsample)
      return 103;

  return chi2;
}

double TPNFit::doFit2( double *adc, double tau1, double tau2, double ampl, 
		       double time, double qmax )
{

  //TFile * file = new TFile("/nfshome0/ecallaser/cmssw/CMSSW_3_2_0_dev2/src/CalibCalorimetry/EcalLaserAnalyzer/test.root","RECREATE");

  ampl2=0.0;
  timeatmax2=0.0;

  htmp->Reset();
  int imax=0; double max=0.;

  for(int is=0; is<NMAXSAMPPN; is++){
    htmp->SetBinContent(is+1,adc[is]);
    if(adc[is]>max){
      max=adc[is];
      imax=is;
    }
  }
  for(int is=0; is<NMAXSAMPPN; is++) htmp->SetBinError(is+1,2.);
    
  fPN->SetParameter(0,ampl);
  fPN->SetParameter(1,time);
  fPN->SetParLimits(0,ampl/2.,ampl*2.);
  fPN->SetParLimits(1,5.,15.);
  fPN->FixParameter(2, tau1);
  fPN->FixParameter(3, tau2);
  fPN->FixParameter(4, qmax);

  // int isup=(int)(time)+fNum_samp_after_max;
  // int iinf=(int)(time)-fNum_samp_bef_max;
  //int isup=(int)(time+10.0);
  //int iinf=(int)(time-tau1/2.0);
  int isup=(int)(imax+10.0);
  int iinf=(int)(imax-tau1/2.0);

  if(isup>50)isup=50;
  if(iinf<5)iinf=5;
  
  int fitStatus= htmp->Fit(fPN,"Q0","",iinf,isup);
  
  ampl2=fPN->GetParameter(0);
  timeatmax2=fPN->GetMaximumX();

  //htmp->Write();
  //file->Close();

  return double(fitStatus);

}

double TPNFit::fitPN_tp(double *x, double *par)
{
  // Convolute SPR with scintillation signal :
  // s(t)=p3*[p0/p1*exp(-(t-p4)/p1)+(1-p0)/p2*exp(-(t-p4)/p2)]

 

  double amp=par[0];
  if(amp<-100.)amp=-100.;
  double t0=par[1];
  double tau1=par[2];
  double tau2=par[3];
  double qmax=par[4];
  if(t0<1.)t0=1.;
  if(t0>25.)t0=25.;
  
  double a=tau2/(tau2-tau1);
  double t=x[0];
  double y=0.;
  
 
  if(t>t0) y= amp*(a*(1.-a)*(TMath::Exp(-(t-t0)/tau1)-TMath::Exp(-(t-t0)/tau2))+
		   (t-t0)/tau1*(1.-a-(t-t0)/2./tau1)*TMath::Exp(-(t-t0)/tau1))/qmax;
 
  return(y);
}

// TF1* TPNFit::funcPN(double *x, double *par){

//   TF1* func = new TF1("fPN",fitPN_tp,0.,double(NMAXSAMPPN),2);
//   return func;
// }

