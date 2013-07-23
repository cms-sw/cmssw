/* 
 *  \class TPNFit
 *
 *  $Date: 2012/02/09 10:08:10 $
 *  \author: Patrice Verrecchia - CEA/Saclay
 */

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TPNFit.h>

#include <iostream>
#include "TVectorD.h"

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

  if(fNsamples > NMAXSAMP2)
          printf("number of PN samples exceed maximum\n");

  for(int k=0;k<NMAXSAMP2;k++)
       t[k]= (double) k;

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
