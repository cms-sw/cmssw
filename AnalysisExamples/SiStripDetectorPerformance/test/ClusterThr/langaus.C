//-----------------------------------------------------------------------
//
//	Convoluted Landau and Gaussian Fitting Function
//         (using ROOT's Landau and Gauss functions)
//
//  Based on a Fortran code by R.Fruehwirth (fruhwirth@hephy.oeaw.ac.at)
//  Adapted for C++/ROOT by H.Pernegger (Heinz.Pernegger@cern.ch) and
//   Markus Friedl (Markus.Friedl@cern.ch)
//
//  to execute this example, do:
//  root > .x langaus.C
// or
//  root > .x langaus.C++
//
//-----------------------------------------------------------------------

#include "TH1.h"
#include "TF1.h"
#include "TROOT.h"
#include "TStyle.h"

Double_t langaufun(Double_t *, Double_t *);

int nFitNum = 0;

Double_t langaufun(Double_t *x, Double_t *par) {

   //Fit parameters:
   //par[0]=Width (scale) parameter of Landau density
   //par[1]=Most Probable (MP, location) parameter of Landau density
   //par[2]=Total area (integral -inf to inf, normalization constant)
   //par[3]=Width (sigma) of convoluted Gaussian function
   //
   //In the Landau distribution (represented by the CERNLIB approximation), 
   //the maximum is located at x=-0.22278298 with the location parameter=0.
   //This shift is corrected within this function, so that the actual
   //maximum is identical to the MP parameter.

      // Numeric constants
      Double_t invsq2pi = 0.3989422804014;   // (2 pi)^(-1/2)
      Double_t mpshift  = -0.22278298;       // Landau maximum location

      // Control constants
      Double_t np = 100.0;      // number of convolution steps
      Double_t sc =   5.0;      // convolution extends to +-sc Gaussian sigmas

      // Variables
      Double_t xx;
      Double_t mpc;
      Double_t fland;
      Double_t sum = 0.0;
      Double_t xlow,xupp;
      Double_t step;
      Double_t i;


      // MP shift correction
      mpc = par[1] - mpshift * par[0]; 

      // Range of convolution integral
      xlow = x[0] - sc * par[3];
      xupp = x[0] + sc * par[3];

      step = (xupp-xlow) / np;

      // Convolution integral of Landau and Gaussian by sum
      for(i=1.0; i<=np/2; i++) {
         xx = xlow + (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);

         xx = xupp - (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);
      }

      return (par[2] * step * sum * invsq2pi / par[3]);
}


TF1 *langaufit(TH1F *his, Double_t *fitrange, Double_t *startvalues, Double_t *parlimitslo, Double_t *parlimitshi, Double_t *fitparams, Double_t *fiterrors, Double_t *ChiSqr, Int_t *NDF,char * opts="RB0")
{
   // Once again, here are the Landau * Gaussian parameters:
   //   par[0]=Width (scale) parameter of Landau density
   //   par[1]=Most Probable (MP, location) parameter of Landau density
   //   par[2]=Total area (integral -inf to inf, normalization constant)
   //   par[3]=Width (sigma) of convoluted Gaussian function
   //
   // Variables for langaufit call:
   //   his             histogram to fit
   //   fitrange[2]     lo and hi boundaries of fit range
   //   startvalues[4]  reasonable start values for the fit
   //   parlimitslo[4]  lower parameter limits
   //   parlimitshi[4]  upper parameter limits
   //   fitparams[4]    returns the final fit parameters
   //   fiterrors[4]    returns the final fit errors
   //   ChiSqr          returns the chi square
   //   NDF             returns ndf

   Int_t i;
   Char_t FunName[100];

   sprintf(FunName,"Fitfcn_%s%d",his->GetName(), nFitNum++);

   TF1 *ffitold = (TF1*)gROOT->GetListOfFunctions()->FindObject(FunName);
   if (ffitold) delete ffitold;

   TF1 *ffit = new TF1(FunName,langaufun,fitrange[0],fitrange[1],4);
   ffit->SetParameters(startvalues);
   ffit->SetParNames("Width","MP","Area","GSigma");
   
   for (i=0; i<4; i++) {
      ffit->SetParLimits(i, parlimitslo[i], parlimitshi[i]);
   }

   his->Fit(FunName,opts);   // fit within specified range, use ParLimits, do not plot

   ffit->GetParameters(fitparams);    // obtain fit parameters
   for (i=0; i<4; i++) {
      fiterrors[i] = ffit->GetParError(i);     // obtain fit parameter errors
   }
   ChiSqr[0] = ffit->GetChisquare();  // obtain chi^2
   NDF[0] = ffit->GetNDF();           // obtain ndf

   return (ffit);              // return fit function

}


Int_t langaupro(Double_t *params, Double_t &maxx, Double_t &FWHM) {

   // Seaches for the location (x value) at the maximum of the 
   // Landau-Gaussian convolute and its full width at half-maximum.
   //
   // The search is probably not very efficient, but it's a first try.

   Double_t p,x,fy,fxr,fxl;
   Double_t step;
   Double_t l,lold;
   Int_t i = 0;
   Int_t MAXCALLS = 10000;


   // Search for maximum

   p = params[1] - 0.1 * params[0];
   step = 0.05 * params[0];
   lold = -2.0;
   l    = -1.0;


   while ( (l != lold) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = langaufun(&x,params);
 
      if (l < lold)
         step = -step/10;
 
      p += step;
   }

   if (i == MAXCALLS)
      return (-1);

   maxx = x;

   fy = l/2;


   // Search for right x location of fy

   p = maxx + params[0];
   step = params[0];
   lold = -2.0;
   l    = -1e300;
   i    = 0;


   while ( (l != lold) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = TMath::Abs(langaufun(&x,params) - fy);
 
      if (l > lold)
         step = -step/10;
 
      p += step;
   }

   if (i == MAXCALLS)
      return (-2);

   fxr = x;


   // Search for left x location of fy

   p = maxx - 0.5 * params[0];
   step = -params[0];
   lold = -2.0;
   l    = -1e300;
   i    = 0;

   while ( (l != lold) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = TMath::Abs(langaufun(&x,params) - fy);
 
      if (l > lold)
         step = -step/10;
 
      p += step;
   }

   if (i == MAXCALLS)
      return (-3);


   fxl = x;

   FWHM = fxr - fxl;
   return (0);
}

void langaus( TH1F *poHist) {
  // Fill Histogram
  printf("Fitting...\n");

  // Setting fit range and start values
  Double_t fr[2];
  Double_t sv[4], pllo[4], plhi[4], fp[4], fpe[4];
  fr[0]=0.3*poHist->GetMean();
  fr[1]=3.0*poHist->GetMean();

  pllo[0]=5.0; pllo[1]=30.0; pllo[2]=1.0; pllo[3]=10.0;
  plhi[0]=25.0; plhi[1]=200.0; plhi[2]=1000000.0; plhi[3]=50.0;
  sv[0]=17.9; sv[1]=100.0; sv[2]=50000.0; sv[3]=42.1;

  Double_t chisqr;
  Int_t    ndf;
  TF1 *fitsnr = langaufit(poHist,fr,sv,pllo,plhi,fp,fpe,&chisqr,&ndf);

  Double_t SNRPeak, SNRFWHM;
  langaupro(fp,SNRPeak,SNRFWHM);

  printf("Fitting done\nPlotting results...\n");

  poHist->Draw( "pe");
  fitsnr->Draw("lsame");
}

void langausN( TH1F *poHist,double start=0, double stop=0) {
  // Fill Histogram
  printf("Fitting...\n");

  // Setting fit range and start values
  Double_t fr[2];
  Double_t sv[4], pllo[4], plhi[4], fp[4], fpe[4];
  fr[0]=0.3*poHist->GetMean();
  fr[1]=3.0*poHist->GetMean();

  if (start<stop){
    fr[0]=start;
    fr[1]=stop;
  }

  pllo[0]=1.0; pllo[1]=4.0; pllo[2]=0.2; pllo[3]=0.2;
  plhi[0]=30.0; plhi[1]=50.0; plhi[2]=200000.0; plhi[3]=10.0;
  sv[0]=15.0; sv[1]=30.0; sv[2]=10000.0; sv[3]=8.0;

  Double_t chisqr;
  Int_t    ndf;
  TF1 *fitsnr = langaufit(poHist,fr,sv,pllo,plhi,fp,fpe,&chisqr,&ndf);

  Double_t SNRPeak, SNRFWHM;
  langaupro(fp,SNRPeak,SNRFWHM);
  std::cout << "SNRPeak " << SNRPeak << std::endl;
  std::cout << "SNRFWHM " << SNRFWHM << std::endl;
  printf("Fitting done\nPlotting results...\n");

  poHist->Draw( "pe");
  fitsnr->Draw("lsame");
}
void langausN( TH1F *poHist,Double_t& SNRPeak,Double_t& SNRFWHM,double start=0, double stop=0,bool draw=false, char* opts="RB0") {
  // Fill Histogram
  printf("Fitting...\n");

  // Setting fit range and start values
  Double_t fr[2];
  Double_t sv[4], pllo[4], plhi[4], fp[4], fpe[4];
  fr[0]=0.3*poHist->GetMean();
  fr[1]=3.0*poHist->GetMean();

  if (start<stop){
    fr[0]=start;
    fr[1]=stop;
  }

  pllo[0]=1.0; pllo[1]=4.0; pllo[2]=0.2; pllo[3]=0.2;
  plhi[0]=30.0; plhi[1]=50.0; plhi[2]=200000.0; plhi[3]=10.0;
  sv[0]=15.0; sv[1]=30.0; sv[2]=10000.0; sv[3]=8.0;

  Double_t chisqr;
  Int_t    ndf;
  TF1 *fitsnr = langaufit(poHist,fr,sv,pllo,plhi,fp,fpe,&chisqr,&ndf,opts);

  //Double_t SNRPeak, SNRFWHM;
  //langaupro(fp,SNRPeak,SNRFWHM);
  
  double fitparams[4];
  fitsnr->GetParameters(fitparams);    // obtain fit parameters

  SNRPeak=fitparams[1];
  SNRFWHM=fitparams[0];
  cout << SNRPeak << " " << SNRFWHM << endl;
  if (draw){
    printf("Fitting done\nPlotting results...\n");

    poHist->Draw( "pe");
    fitsnr->Draw("lsame");
  }
}

