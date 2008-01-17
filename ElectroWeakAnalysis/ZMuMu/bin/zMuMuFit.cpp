#include "ElectroWeakAnalysis/ZMuMu/interface/ConvZShapeGauss.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/Gaussian.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/Exponential.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/Polinomial.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/ZetaShape.h"
#include "TMinuit.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TF1.h"
#include "TStyle.h"
#include <iostream>
#include <cmath>

using namespace std;

static const int fitXMin = 80; 
static const int fitXMax = 104;
// assuming binning : Zmm = 200; Zmt = 100; Zms = 50; than 
static const int BinWidthZmm = 1;
static const int BinWidthZmt = 2;
static const int BinWidthZms = 4;
unsigned int nPars = 17;

static size_t fullBins = 0;

TF1 * fitZmm;
TF1 * fitZmt;  
TF1 * fitZms;
TF1 * bkgZmt;
TF1 * fitZmt_sig;

// Functions used for integral error computation

double df_dParmm(double * x, double * p) { 
   // derivative of the function w.r..t parameters
   // use calculated derivatives from TF1::GradientPar

   double grad[17]; 
   // p is used to specify for which parameter the derivative is computed 
   int ipar = int(p[0] ); 
   fitZmm->GradientPar(x, grad);

   return grad[ipar]; 
}

double df_dParmt(double * x, double * p) { 
   // derivative of the function w.r..t parameters
   // use calculated derivatives from TF1::GradientPar

   double grad[17]; 
   // p is used to specify for which parameter the derivative is computed 
   int ipar = int(p[0] ); 
   fitZmt->GradientPar(x, grad);

   return grad[ipar]; 
}

double df_dParms(double * x, double * p) { 
   // derivative of the function w.r..t parameters
   // use calculated derivatives from TF1::GradientPar

   double grad[17]; 
   // p is used to specify for which parameter the derivative is computed 
   int ipar = int(p[0] ); 
   fitZms->GradientPar(x, grad);

   return grad[ipar]; 
}

double df_dParbkg(double * x, double * p) { 
   // derivative of the function w.r..t parameters
   // use calculated derivatives from TF1::GradientPar

   double grad[4]; 
   // p is used to specify for which parameter the derivative is computed 
   int ipar = int(p[0] ); 
   bkgZmt->GradientPar(x, grad);

   return grad[ipar]; 
}

double df_dParmtsig(double * x, double * p) { 
  // derivative of the function w.r..t parameters
  // use calculated derivatives from TF1::GradientPar
  
  double grad[13]; 
  // p is used to specify for which parameter the derivative is computed 
  int ipar = int(p[0] ); 
  fitZmt_sig->GradientPar(x, grad);
  
  return grad[ipar]; 
}

double BWZmm (double *x, double * p) {
  // x[0] = E ; sp = E'^2 ; y = E'; H(E) = /int{BW(E') * g(E-E') dE'}
  //Ni = p[4]; Nf = p[3]; m = p[1]; g = p[2];
  // double * p = { &m, &g, &Nf, &Ni};
  //sigma = p[7]; mean = p[8];
  static const int bins = 100;
  ZetaShape zs(p[1], p[2], p[3], p[4]);
  Gaussian gauss(p[8], p[7]);
  double DeltaX = 2*gauss.confidenceLevel99_7Interval();
  ConvZShapeGauss czg(zs, gauss, DeltaX, bins);
  double BWG = czg(x[0]);
  //l = p[18]; a = p[19]; b = p[20]; pol = (-l*(l*a*x+a+l)-b*(l^2*x^2+2*l*x+2));
  const double a0 = -pow(p[18], 2)*p[20];
  const double a1 = -pow(p[18], 2)*p[19]-2*p[18]*p[20];
  const double a2 = -pow(p[18], 2)-p[18]*p[19]-2*p[20];
  Exponential expo(-p[18]);
  Polinomial pol(a0, a1, a2); 
  //double N1 = expo(fitXMax)/pow(p[18], 3)*pol(fitXMax);
  //double N2 = expo(fitXMin)/pow(p[18], 3)*pol(fitXMin);
  // return  BWG * p[0] /(10) * .946  * p[5] * p[5] * p[6] * p[6] +  p[17]/(N1-N2) * exp( -l * x[0] )  * (1 + x[0] * ( a + x[0] * b));
  return  BWG * p[0] * p[5] * p[5] * p[6] * p[6] ;
  // +  p[17]/(N1-N2) * exp( -l * x[0] )  * (1 + x[0] * ( a + x[0] * b));
}

double BWZmt (double *x, double * p) {
  // x[0] = E ; sp = E'^2 ; y = E'; H(E) = /int{BW(E') * g(E-E') dE'}
  //Ni = p[4]; Nf = p[3]; m = p[1]; double g = p[2];
  // double * p = { &m, &g, &Nf, &Ni};
  //sigma = p[9]; mean = p[10];
  static const int bins = 100;
  ZetaShape zs(p[1], p[2], p[3], p[4]);
  Gaussian gauss(p[10], p[9]);
  double DeltaX = 2*gauss.confidenceLevel99_7Interval();
  ConvZShapeGauss czg(zs, gauss, DeltaX, bins);
  double BWZ = czg(x[0]);
  //l = p[14]; a = p[15]; b = p[16]; pol = (-l*(l*a*x+a+l)-b*(l^2*x^2+2*l*x+2));
  const double a0 = -pow(p[14], 2)*p[16];
  const double a1 = -pow(p[14], 2)*p[15]-2*p[14]*p[16];
  const double a2 = -pow(p[14], 2)-p[14]*p[15]-2*p[16];
  Exponential expo(-p[14]);
  Polinomial pol(a0, a1, a2); 
  double N1 = expo(fitXMax)/pow(p[14], 3)*pol(fitXMax);
  double N2 = expo(fitXMin)/pow(p[14], 3)*pol(fitXMin);
  Polinomial poli(p[16], p[15], 1);
  return BinWidthZmt  * p[0] * BWZ * 2 * (p[5]* p[5] * p[6]* (1.0-p[6])) +  p[13]/(N1-N2) * expo(x[0])  * poli(x[0]);
}

double BWZms (double *x, double * p) {
  // x[0] = E ; sp = E'^2 ; y = E'; H(E) = /int{BW(E') * g(E-E') dE'}
  //Ni = p[4]; Nf = p[3]; m = p[1]; g = p[2];
  //double * p = { &m, &g, &Nf, &Ni};
  //sigma = p[11]; mean = p[12];
  static const int bins = 100;
  ZetaShape zs(p[1], p[2], p[3], p[4]);
  Gaussian gauss(p[12], p[11]);
  double DeltaX = 2*gauss.confidenceLevel99_7Interval();
  ConvZShapeGauss czg(zs, gauss, DeltaX, bins);
  double BWZ = czg(x[0]);
  /*
  double l = p[22];
  double a = p[23];
  double b = p[24];
  
  double N1 = exp (- l * fitXMax) / (l* l * l) * (-l * (l * fitXMax * a + a + l) - b * (l * l * fitXMax * fitXMax +  2 * l * fitXMax + 2 ));
  
  double N2 = exp (- l * fitXMin) / (l* l * l) * (-l * (l * fitXMin * a + a + l) - b * (l * l * fitXMin * fitXMin + 2 * l  * fitXMin + 2 ));
  */
  return BinWidthZms * p[0] * BWZ *2* (p[6]* p[6] * p[5]* (1.0-p[5])) 
    //+  p[21]/(N1-N2) * exp( -l * x[0] )  * (1 + x[0] * ( a + x[0] * b)) 
;
}

// Function used for computing signal integral and error in Zmt case
double BWZmt_sig (double *x, double * p) {
  // sigma = p[9]; mean = p[10];
  static const int bins = 100;
  ZetaShape zs(p[1], p[2], p[3], p[4]);
  Gaussian gauss(p[10], p[9]);
  double DeltaX = 2*gauss.confidenceLevel99_7Interval();
  ConvZShapeGauss czg(zs, gauss, DeltaX, bins);
  double BWZ = czg(x[0]);
  return BinWidthZmt * p[0] * BWZ *2* (p[5]* p[5] * p[6]* (1.0-p[6])) ;
}

double pol2(double *x,double *p){
  //l = p[1]; a = p[2]; b = p[3]; pol = (-l*(l*a*x+a+l)-b*(l^2*x^2+2*l*x+2));
  const double a0 = -pow(p[1], 2)*p[3];
  const double a1 = -pow(p[1], 2)*p[2]-2*p[1]*p[3];
  const double a2 = -pow(p[1], 2)-p[1]*p[2]-2*p[3];
  Exponential expo(-p[1]);
  Polinomial pol(a0, a1, a2);
  double N1 = expo(fitXMax)/pow(p[1], 3)*pol(fitXMax);
  double N2 = expo(fitXMin)/pow(p[1], 3)*pol(fitXMin);
  Polinomial poli(p[3], p[2], 1);
  return p[0]/(N1-N2) * expo(x[0]) * poli(x[0]);
}

void fcn( int &, double *, double & f, double * par, int ) {
  static bool firstTime = true;
  static const size_t nBins = 200; 
  static const size_t n = nBins, nn = nBins/BinWidthZmt, nnn = nBins/BinWidthZms;
  static double h[n] , dh[n];
  static double hh[nn], dhh[nn];
  static double hhh[nnn], dhhh[nnn];
  static double xMin, xMax, deltaX, ddeltaX, dddeltaX;
  if ( firstTime ) {
    firstTime = false;
    //    gROOT->ProcessLine(".L fit_function.C+");
    
    
    //Zmm senza back
    TFile * ZToLL_file1 = new TFile("../test/ZMM_ZMT_ZMS_histo/ZCandidates_Histo_iso_all.root","read");
    TH1D * zToMuMu = (TH1D*) ZToLL_file1->Get("zToMM");
    TH1D * zToSingleTrackMu = (TH1D*) ZToLL_file1->Get("zToMTk");
    TH1D * zToSingleStandAloneMu = (TH1D*) ZToLL_file1->Get("zToMS");
    
    // zToSingleTrackMu->Rebin(2);
    // zToSingleStandAloneMu->Rebin(4);
    
    
    xMin = zToMuMu->GetXaxis()->GetXmin();
    xMax = zToMuMu->GetXaxis()->GetXmax();
    
    cout << xMin << "  " << xMax << endl;
      double delta = xMax - xMin;
      deltaX = delta / n;
      ddeltaX = delta / nn;
      dddeltaX = delta / nnn;
      
      for( size_t i = 0; i < n; ++i ) {
	h[i] = zToMuMu->GetBinContent(i+1);
	dh[i] = zToMuMu->GetBinError(i+1);
	double x = xMin + ( i +.5 ) * deltaX;
	//      cout << "fitXMin " << fitXMin << " x " << x << " fitXMax " << fitXMax << " h[i] " << h[i] << endl;
	if( (x > fitXMin) && (x < fitXMax) && ( h[i]>0 ) ) 
	  fullBins++; 
	//cout << "fullbinsMM = " << fullBins << endl; 
      }
      
      for( size_t i = 0; i < nn; ++i ) {
	hh[i] = zToSingleTrackMu->GetBinContent(i+1);
	dhh[i] = zToSingleTrackMu->GetBinError(i+1);
	double x = xMin + ( i +.5 ) * ddeltaX;
	//      if( x > fitXMin && x < fitXMax &&  (hh[i]>0)) fullBins =fullBins++; 
	if( x > fitXMin && x < fitXMax &&  (hh[i]>0)) 
	  fullBins++; 
	}
      
      //cout << "fullbinsMT = " << fullBins << endl; 
      
      for( size_t i = 0; i < nnn; ++i ) {
	hhh[i] = zToSingleStandAloneMu->GetBinContent(i+1);
	dhhh[i] = zToSingleStandAloneMu->GetBinError(i+1);
	double x = xMin + ( i  + .5) * dddeltaX;
	//      if( x > fitXMin && x < fitXMax &&  (hhh[i]>0)) fullBins =fullBins++; 
	if( x > fitXMin && x < fitXMax &&  (hhh[i]>0)) 
	  fullBins++; 
      }
      //cout << "fullbinsMS = " << fullBins << endl; 
      
  }
  static double var[1];
  double f1 =0, f2=0, f3=0;
  f=0; 
  //cout << " f = " << f;
  for( size_t i = 0; i < n; ++ i ) {
    double x = xMin + ( i + .5 ) * deltaX;
    if ( x > fitXMin && x < fitXMax && h[i] > 0    ) {
      var[0] = x; 
   double l = BWZmm( var, par );
      double delta = l- h[i];
      // f += (delta * delta) / (l) ;    
        f1 += (delta * delta) / (dh[i]* dh[i]) ;
      
    }
  }
  //cout << " --> " << f1;
  for( size_t i = 0; i < nn; ++ i ) {
    double x = xMin + ( i +.5 ) * ddeltaX;
    if ( x > fitXMin && x < fitXMax && hh[i]>0   ) {
      var[0] = x; 
      double ll = BWZmt( var, par ) ;
      double ddelta = ll - hh[i];
      //      cout << "ll = " << ll << " hh[i] = " << hh[i] << " dhh[i] = " << dhh[i] << " nRebinZmt = " << nRebinZmt << endl;
      //f2 += (ddelta * ddelta) / ( nRebinZmt/10. * dhh[i] * dhh[i]) ;
      f2 += (ddelta * ddelta) / ( dhh[i] * dhh[i]) ;
     
    }
  }
  //cout << " --> " << f2;  
  for( size_t i = 0; i < nnn; ++ i ) {
    double x = xMin + ( i +.5) * dddeltaX;
    if ( x > fitXMin && x < fitXMax &&  hhh[i]>0 ) {
      var[0] = x; 
      double lll = BWZms( var, par );
      double dddelta = lll- hhh[i];
      //      cout << "lll = " << lll << " hhh[i] = " << hhh[i] << " dhhh[i] = " << dhhh[i] << endl;
   //f3 += (dddelta * dddelta) / (nRebinZms/10. * hhh[i])  ;     
      f3 += (dddelta * dddelta) / (hhh[i])  ;     
    }
  }
  //cout << " --> " << f3 << endl;
  f = f1 + f2 + f3;
  //cout << "f  === " << f << endl;
  
}
 
void zMinuit3Independentfit_conv() {
  
  cout << ">>> starting fit program" << endl;


  cout << ">>> fit init" << endl;
  
  unsigned int nPars = 17;
  TMinuit minuit( nPars );
  minuit.SetFCN( fcn );
  int ierflg = 0;
  double arglist[ 10 ];
  arglist[0] = -1;
  minuit.mnexcm( "SET PRINT", arglist, 1, ierflg );
  if ( ierflg != 0 ) cerr << "set print: AARGH!!" << endl;
  arglist[0] = 1;
  minuit.mnexcm( "SET ERR", arglist, 1, ierflg );
  if ( ierflg != 0 ) cerr << "set err: AARGH!!" << endl;
  // set parameters
  string names[] = { "P0", "mZmm", "GZmm", 
		     "f_gamma", "f_int", 
		     "eff_track","eff_standalone", "sigmaZmm", "meanZmm","sigmaZmt", "meanZmt", "sigmaZms", "meanZms","f_backZmt", "l_backZmt", "a_backZmt", "b_backZmt" 
		     //, "f_backZmm", "l_backZmm", "a_backZmm", "b_backZmm"
		     //, "f_backZms", "l_backZms", "a_backZms", "b_backZms"
};
  

  cout << ">>> set fit parms" << endl;

  minuit.mnparm( 0, names[0].c_str(),6000  ,10 , 100, 100000, ierflg ); 
  
  //  minuit.FixParameter( 0 );
  minuit.mnparm( 1, names[1].c_str(), 91.3, .1, 70., 110, ierflg );
  // minuit.FixParameter( 1 );
  minuit.mnparm( 2, names[2].c_str(), 2.5, 1, 1, 10, ierflg ); 
  //minuit.FixParameter( 2 ); 
  minuit.mnparm( 3, names[3].c_str(), 0, 0.1, -100, 1000, ierflg ); 
  minuit.FixParameter( 3 );
  minuit.mnparm( 4, names[4].c_str(),.001, .0001, -1000000, 1000000, ierflg ); 
  //minuit.FixParameter( 4 );
  minuit.mnparm( 5, names[5].c_str(), 0.98, .01, 0.,1, ierflg ); 
  //  minuit.FixParameter( 5 );
  minuit.mnparm( 6, names[6].c_str(), 0.97, .01, 0., 1,ierflg ); 
  //minuit.FixParameter( 6 );
  minuit.mnparm( 7, names[7].c_str(), 1., .1, -5, 5, ierflg ); 
  // minuit.FixParameter( 7);
  minuit.mnparm( 8, names[8].c_str(), 0, .001, -.5, .5, ierflg ); 
  minuit.FixParameter( 8 );


 minuit.mnparm( 9, names[9].c_str(), 1, .1, -3, 3, ierflg ); 
 //minuit.FixParameter( 9);
 minuit.mnparm( 10, names[10].c_str(), 1., .1, -10, 10, ierflg ); 
 //minuit.FixParameter( 10);
 minuit.mnparm( 11, names[11].c_str(), 5, .1, -15, 15, ierflg ); 
 minuit.mnparm( 12, names[12].c_str(), 0., .1, -1, 1, ierflg ); 
  minuit.mnparm( 13, names[13].c_str(), 400, 1, 0, 10000, ierflg ); 
  minuit.mnparm( 14, names[14].c_str(), 0.01,  .001, 0, 100, ierflg ); 
  minuit.mnparm( 15, names[15].c_str(), 0, 1, -10000, 10000, ierflg ); 
  minuit.FixParameter( 15 );
  minuit.mnparm( 16, names[16].c_str(), 0, 1, -1000, 1000, ierflg ); 
  minuit.FixParameter( 15 );
  /*
  minuit.mnparm( 17, names[17].c_str(), 0., .1, 0, 60, ierflg ); 
  minuit.FixParameter( 17 );
 
  minuit.mnparm( 18, names[18].c_str(), 0.01,  0.001,0., 0.1, ierflg ); 
  minuit.mnparm( 19, names[19].c_str(), 100, .1, 0, 10000, ierflg ); 
  minuit.mnparm( 20, names[20].c_str(), 0, .0001, -1000, 1000, ierflg ); 
  minuit.FixParameter( 20);
  
    minuit.mnparm( 21, names[21].c_str(), 50, 1, -10, 100000, ierflg ); 
    minuit.mnparm( 22, names[22].c_str(), 0.0001,  .0001, -100, 100, ierflg ); 
    minuit.mnparm( 23, names[23].c_str(), 5, 1, -10000, 10000, ierflg ); 
    minuit.mnparm( 24, names[24].c_str(), .1, 1, -1000, 1000, ierflg ); 
  */
  cout << ">>> run fit" << endl;
  
  arglist[0] = 5000;
  arglist[1] = .1;
  minuit.mnexcm("MIGRAD",arglist, 2, ierflg );
  if ( ierflg != 0 ) cerr << "migrad: AARGH!!" << endl;
  // minuit.Migrad();
  // minuit.mnmnos();
  minuit.mnmatu(1);
  
  cout << ">>> fit completed" << endl;

  double amin;
  double edm, errdef;
  int nvpar, nparx;
  minuit.mnstat( amin, edm, errdef, nvpar, nparx, ierflg );  
  minuit.mnprin( 3, amin );
  unsigned int ndof = fullBins - minuit.GetNumFreePars();
  cout << "Chi-2 = " << amin << "/" << ndof << " = " << amin/ndof 
       << "; prob: " << TMath::Prob( amin, ndof )
       << endl;
  //minuit.mnmatu(1);
 
   
  //   TF1 * fitZmm = new TF1("fitZmm",BWZmm,fitXMin,fitXMax,nPars); 
  //   TF1 * fitZmt = new TF1("fitZmt",BWZmt,fitXMin,fitXMax,nPars); 
  //   TF1 * fitZms = new TF1("fitZms",BWZms,fitXMin,fitXMax,nPars); 

   fitZmm = new TF1("fitZmm",BWZmm,fitXMin,fitXMax,nPars); 
   fitZmt = new TF1("fitZmt",BWZmt,fitXMin,fitXMax,nPars); 
   fitZms = new TF1("fitZms",BWZms,fitXMin,fitXMax,nPars); 
   fitZmm->SetLineColor(kRed);
   fitZmt->SetLineColor(kRed);
   fitZms->SetLineColor(kRed);
   TF1 * bkgZmm = new TF1("bkgZmm",pol2,fitXMin,fitXMax,4);
   TF1 * bkgZms = new TF1("bkgZms",pol2,fitXMin,fitXMax,4);
   //   TF1 * bkgZmt = new TF1("bkgZmt",pol2,fitXMin,fitXMax,4);
   bkgZmt = new TF1("bkgZmt",pol2,fitXMin,fitXMax,4);

   // Signal-only function in the Zmt case
   fitZmt_sig = new TF1("fitZmt_sig",BWZmt_sig,fitXMin,fitXMax,13); 
   
   
   gStyle->SetOptFit(1111);
   // gStyle->SetOptLogy();
   
   for( size_t i = 0; i < nPars; ++ i ) {
     double p, dp;
     minuit.GetParameter( i, p, dp );
     cout << names[i] << " = " << p << "+/-" << dp << endl;
   }
   
   for( size_t i = 0; i < nPars; ++ i ) {   
     double p, dp;
     minuit.GetParameter( i, p, dp );
     fitZmm->SetParameter(i,p);
     //bkgZmm->SetParameter(j,0.);
   }
   
   for( size_t i = 0; i < nPars; ++ i ) {   
     double p, dp;
     minuit.GetParameter( i, p, dp );
     fitZmt->SetParameter(i,p);
     //bkgZmt->SetParameter(i,0.);
   }
   
   for( size_t i = 0; i < nPars; ++ i ) {   
     double p, dp;
     minuit.GetParameter( i, p, dp );
     fitZms->SetParameter(i,p);
   }
    
  for( size_t i = 13; i < 17; ++ i ) {   
  double p, dp;
  minuit.GetParameter( i, p, dp );
  bkgZmt->SetParameter(i-13,p);
}
for( size_t i = 17; i < 21; ++ i ) {   
  double p, dp;
  minuit.GetParameter( i, p, dp );
  bkgZmm->SetParameter(i-17,p);
}


// Setting signal-only function in the Zmt case
 for( size_t i = 0; i < 13; ++ i ) {   
   double p, dp;
   minuit.GetParameter( i, p, dp );
   fitZmt_sig->SetParameter(i,p);
 }
   

/*
for( size_t i = 21; i < 25; ++ i ) {   
  double p, dp;
  minuit.GetParameter( i, p, dp );
  bkgZms->SetParameter(i-21,p);
}
*/

 TFile * ZToLL_file1 = new TFile("../test/ZMM_ZMT_ZMS_histo/ZCandidates_Histo_iso_all.root","read");
 TH1D * zToMuMu = (TH1D*) ZToLL_file1->Get("zToMM");
 TH1D * zToSingleTrackMu = (TH1D*) ZToLL_file1->Get("zToMTk");
 
 TH1D * zToSingleStandAloneMu = (TH1D*) ZToLL_file1->Get("zToMS");
 //  zToSingleTrackMu->Rebin(2);
 
 //  zToSingleStandAloneMu->Rebin(4);
 gStyle->SetOptLogy();  
 TCanvas c;
  
    zToMuMu->Draw("e");
    //zToMuMu->GetXaxis()->SetRangeUser(85,97);

    zToMuMu->GetXaxis()->SetRangeUser(fitXMin-10, fitXMax+10);
    zToMuMu->SetXTitle("#mu #mu mass (GeV/c^{2})");
    zToMuMu->SetTitle("fit of #mu #mu mass with isolation cut");
    zToMuMu->SetYTitle("Events / 0.1 GeV/c^{2}");


   /* 
  TH1D * h = new TH1D("h","h", 2000 , 80, 100);
   for (int i = 0; i < h->GetNbinsX(); i ++) {
     h->SetBinContent(i,zToMuMu->GetBinContent(8000 + i) );
     h->SetBinError(i,zToMuMu->GetBinError(8000 + i) ) ; 
      }
      // bkgZmm->Draw("same");
  
   //   fitZmm->GetXaxis()->SetLimits(fitXMin,fitXMax);
   h->Draw("e");
   */
   fitZmm->Draw("same");
   bkgZmm->Draw("same");
   c.SaveAs("zmm.eps");

 
   TCanvas c1;
   
   zToSingleTrackMu->Draw("e");

zToSingleTrackMu->GetXaxis()->SetRangeUser(fitXMin-20, fitXMax+20);
   zToSingleTrackMu->SetXTitle("#mu + (unmatched) track mass (GeV/c^{2})");
    zToSingleTrackMu->SetTitle("fit of #mu + (unmatched) track  mass with isolation cut");
    zToSingleTrackMu->SetYTitle("Events / 2 GeV/c^{2}");
   //  bkgZmt->Draw("same");
   fitZmt->Draw("same");
    bkgZmt->Draw("same");
  c1.SaveAs( "zmt.eps");
   
   TCanvas c2;
   
   zToSingleStandAloneMu->Draw("e");
zToSingleStandAloneMu->GetXaxis()->SetRangeUser(fitXMin-20, fitXMax+20);
    zToSingleStandAloneMu->SetXTitle("#mu + (unmatched) standalone mass (GeV/c^{2})");
    zToSingleStandAloneMu->SetTitle("fit of #mu + (unmatched) standalone  mass with isolation cut");
    zToSingleStandAloneMu->SetYTitle("Events / 4 GeV/c^{2}");
    //bkgZms->Draw("same");
   fitZms->Draw("same");
   c2.SaveAs( "zms.eps");
   
     //   fitZmm->SetParameter(4,0);
   //double p4, dp4;
   //minuit.GetParameter( 4, p4, dp4 );
   //   cout <<"Integral of Zmm == "<<fitZmm->Integral(fitXMin ,fitXMax)<<endl;
   double Int = fitZmm->Integral(fitXMin,fitXMax);
   int scale =  (int) (1./BinWidthZmm);
   cout <<"scale == "<<scale<<endl;
   double IntHisto = zToMuMu->Integral((scale *fitXMin)+1, (scale* fitXMax)+1);

   cout <<"Integral of Zmm == "<<Int<<"vs Histo Integral "<<IntHisto<<endl;   

 double p0, dp0;
   minuit.GetParameter( 0, p0, dp0 );
   double p5, dp5;
   minuit.GetParameter( 5, p5, dp5 );
   double p6, dp6;
   minuit.GetParameter( 6, p6, dp6 );
   double p17, dp17;
   minuit.GetParameter( 17, p17, dp17 );

   // double Norm  =   Int/ p0;
   //  double Norm = 
   // cout <<"Norm == "<<Norm <<endl;
    
   fitZmm->SetParameter(17,0);


   // Now computing yields with errors in the fitted range
   // YIELDS FOR ZToMuMu
   double IntS = fitZmm->Integral(fitXMin ,fitXMax);
   cout <<"N_Zmm = " << IntS << " +/- " ;
   
   //  double ErrorIntS = fitZmm->IntegralError(fitXMin ,fitXMax);
   //   cout <<"Error_N_Zmm = " << ErrorIntS <<endl; 

   // Computing the error on the integral (no correlation hyp.)
   TF1 * derivMM_par0 = new TF1("dfdp0",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par0->SetParameter(0,0);
   TF1 * derivMM_par1 = new TF1("dfdp1",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par1->SetParameter(0,1.);
   TF1 * derivMM_par2 = new TF1("dfdp2",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par2->SetParameter(0,2.);
   TF1 * derivMM_par3 = new TF1("dfdp3",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par3->SetParameter(0,3.);
   TF1 * derivMM_par4 = new TF1("dfdp4",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par4->SetParameter(0,4.);
   TF1 * derivMM_par5 = new TF1("dfdp5",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par5->SetParameter(0,5.);
   TF1 * derivMM_par6 = new TF1("dfdp6",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par6->SetParameter(0,6.);
   TF1 * derivMM_par7 = new TF1("dfdp7",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par7->SetParameter(0,7.);
   TF1 * derivMM_par8 = new TF1("dfdp8",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par8->SetParameter(0,8.);
   TF1 * derivMM_par9 = new TF1("dfdp9",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par9->SetParameter(0,9.);
   TF1 * derivMM_par10 = new TF1("dfdp10",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par10->SetParameter(0,10.);
   TF1 * derivMM_par11 = new TF1("dfdp11",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par11->SetParameter(0,11.);
   TF1 * derivMM_par12 = new TF1("dfdp12",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par12->SetParameter(0,12.);
   TF1 * derivMM_par13 = new TF1("dfdp13",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par13->SetParameter(0,13.);
   TF1 * derivMM_par14 = new TF1("dfdp14",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par14->SetParameter(0,14.);
   TF1 * derivMM_par15 = new TF1("dfdp15",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par15->SetParameter(0,15.);
   TF1 * derivMM_par16 = new TF1("dfdp16",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par16->SetParameter(0,16.);
   TF1 * derivMM_par17 = new TF1("dfdp17",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par17->SetParameter(0,17.);
   TF1 * derivMM_par18 = new TF1("dfdp18",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par18->SetParameter(0,18.);
   TF1 * derivMM_par19 = new TF1("dfdp19",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par19->SetParameter(0,19.);
   TF1 * derivMM_par20 = new TF1("dfdp20",df_dParmm,fitXMin,fitXMax,1);
   derivMM_par20->SetParameter(0,20.);

   double coe[21];

   coe[0] = derivMM_par0->Integral(fitXMin,fitXMax); 
   coe[1] = derivMM_par1->Integral(fitXMin,fitXMax); 
   coe[2] = derivMM_par2->Integral(fitXMin,fitXMax); 
   coe[3] = derivMM_par3->Integral(fitXMin,fitXMax); 
   coe[4] = derivMM_par4->Integral(fitXMin,fitXMax); 
   coe[5] = derivMM_par5->Integral(fitXMin,fitXMax); 
   coe[6] = derivMM_par6->Integral(fitXMin,fitXMax); 
   coe[7] = derivMM_par7->Integral(fitXMin,fitXMax); 
   coe[8] = derivMM_par8->Integral(fitXMin,fitXMax); 
   coe[9] = derivMM_par9->Integral(fitXMin,fitXMax); 
   coe[10] = derivMM_par10->Integral(fitXMin,fitXMax); 
   coe[11] = derivMM_par11->Integral(fitXMin,fitXMax); 
   coe[12] = derivMM_par12->Integral(fitXMin,fitXMax); 
   coe[13] = derivMM_par13->Integral(fitXMin,fitXMax); 
   coe[14] = derivMM_par14->Integral(fitXMin,fitXMax); 
   coe[15] = derivMM_par15->Integral(fitXMin,fitXMax); 
   coe[16] = derivMM_par16->Integral(fitXMin,fitXMax); 
   coe[17] = derivMM_par17->Integral(fitXMin,fitXMax); 
   coe[18] = derivMM_par18->Integral(fitXMin,fitXMax); 
   coe[19] = derivMM_par19->Integral(fitXMin,fitXMax); 
   coe[20] = derivMM_par20->Integral(fitXMin,fitXMax); 

   double erri2MM = 0;

   for(int i = 0; i<nPars; i++){
     double p, dp;
     minuit.GetParameter( i, p, dp );
     erri2MM += coe[i] * coe[i] * dp * dp;
   }

   cout << sqrt(erri2MM) << endl;



   // YIELDS FOR ZToMuTrk
   double IntSMT = ( fitZmt_sig->Integral(fitXMin ,fitXMax) ) / 2.0;
   cout <<"N_Zmt = " << IntSMT << " +/- " ;

   TF1 * derivMT_par0 = new TF1("dfdp0",df_dParmtsig,fitXMin,fitXMax,1);
   derivMT_par0->SetParameter(0,0);
   TF1 * derivMT_par1 = new TF1("dfdp1",df_dParmtsig,fitXMin,fitXMax,1);
   derivMT_par1->SetParameter(0,1.);
   TF1 * derivMT_par2 = new TF1("dfdp2",df_dParmtsig,fitXMin,fitXMax,1);
   derivMT_par2->SetParameter(0,2.);
   TF1 * derivMT_par3 = new TF1("dfdp3",df_dParmtsig,fitXMin,fitXMax,1);
   derivMT_par3->SetParameter(0,3.);
   TF1 * derivMT_par4 = new TF1("dfdp4",df_dParmtsig,fitXMin,fitXMax,1);
   derivMT_par4->SetParameter(0,4.);
   TF1 * derivMT_par5 = new TF1("dfdp5",df_dParmtsig,fitXMin,fitXMax,1);
   derivMT_par5->SetParameter(0,5.);
   TF1 * derivMT_par6 = new TF1("dfdp6",df_dParmtsig,fitXMin,fitXMax,1);
   derivMT_par6->SetParameter(0,6.);
   TF1 * derivMT_par7 = new TF1("dfdp7",df_dParmtsig,fitXMin,fitXMax,1);
   derivMT_par7->SetParameter(0,7.);
   TF1 * derivMT_par8 = new TF1("dfdp8",df_dParmtsig,fitXMin,fitXMax,1);
   derivMT_par8->SetParameter(0,8.);
   TF1 * derivMT_par9 = new TF1("dfdp9",df_dParmtsig,fitXMin,fitXMax,1);
   derivMT_par9->SetParameter(0,9.);
   TF1 * derivMT_par10 = new TF1("dfdp10",df_dParmtsig,fitXMin,fitXMax,1);
   derivMT_par10->SetParameter(0,10.);
   TF1 * derivMT_par11 = new TF1("dfdp11",df_dParmtsig,fitXMin,fitXMax,1);
   derivMT_par11->SetParameter(0,11.);
   TF1 * derivMT_par12 = new TF1("dfdp12",df_dParmtsig,fitXMin,fitXMax,1);
   derivMT_par12->SetParameter(0,12.);

   coe[0] = derivMT_par0->Integral(fitXMin,fitXMax); 
   coe[1] = derivMT_par1->Integral(fitXMin,fitXMax); 
   coe[2] = derivMT_par2->Integral(fitXMin,fitXMax); 
   coe[3] = derivMT_par3->Integral(fitXMin,fitXMax); 
   coe[4] = derivMT_par4->Integral(fitXMin,fitXMax); 
   coe[5] = derivMT_par5->Integral(fitXMin,fitXMax); 
   coe[6] = derivMT_par6->Integral(fitXMin,fitXMax); 
   coe[7] = derivMT_par7->Integral(fitXMin,fitXMax); 
   coe[8] = derivMT_par8->Integral(fitXMin,fitXMax); 
   coe[9] = derivMT_par9->Integral(fitXMin,fitXMax); 
   coe[10] = derivMT_par10->Integral(fitXMin,fitXMax); 
   coe[11] = derivMT_par11->Integral(fitXMin,fitXMax); 
   coe[12] = derivMT_par12->Integral(fitXMin,fitXMax); 

   double erri2MT = 0;
   
   for(int i = 0; i<13; i++){
     double p, dp;
     minuit.GetParameter( i, p, dp );
     erri2MT += coe[i] * coe[i] * dp * dp;
   }

   cout << sqrt(erri2MT)/2.0 << endl;
   //   cout << "ERROR_Nmt = " << sqrt(erri2MT)/2.0 << endl;

   // YIELDS FOR ZToMuSa
   double IntSMS = ( fitZms->Integral(fitXMin ,fitXMax) ) / 4.0;
   cout <<"N_Zms = " << IntSMS << " +/- " ;

   TF1 * derivMS_par0 = new TF1("dfdp0",df_dParms,fitXMin,fitXMax,1);
   derivMS_par0->SetParameter(0,0);
   TF1 * derivMS_par1 = new TF1("dfdp1",df_dParms,fitXMin,fitXMax,1);
   derivMS_par1->SetParameter(0,1.);
   TF1 * derivMS_par2 = new TF1("dfdp2",df_dParms,fitXMin,fitXMax,1);
   derivMS_par2->SetParameter(0,2.);
   TF1 * derivMS_par3 = new TF1("dfdp3",df_dParms,fitXMin,fitXMax,1);
   derivMS_par3->SetParameter(0,3.);
   TF1 * derivMS_par4 = new TF1("dfdp4",df_dParms,fitXMin,fitXMax,1);
   derivMS_par4->SetParameter(0,4.);
   TF1 * derivMS_par5 = new TF1("dfdp5",df_dParms,fitXMin,fitXMax,1);
   derivMS_par5->SetParameter(0,5.);
   TF1 * derivMS_par6 = new TF1("dfdp6",df_dParms,fitXMin,fitXMax,1);
   derivMS_par6->SetParameter(0,6.);
   TF1 * derivMS_par7 = new TF1("dfdp7",df_dParms,fitXMin,fitXMax,1);
   derivMS_par7->SetParameter(0,7.);
   TF1 * derivMS_par8 = new TF1("dfdp8",df_dParms,fitXMin,fitXMax,1);
   derivMS_par8->SetParameter(0,8.);
   TF1 * derivMS_par9 = new TF1("dfdp9",df_dParms,fitXMin,fitXMax,1);
   derivMS_par9->SetParameter(0,9.);
   TF1 * derivMS_par10 = new TF1("dfdp10",df_dParms,fitXMin,fitXMax,1);
   derivMS_par10->SetParameter(0,10.);
   TF1 * derivMS_par11 = new TF1("dfdp11",df_dParms,fitXMin,fitXMax,1);
   derivMS_par11->SetParameter(0,11.);
   TF1 * derivMS_par12 = new TF1("dfdp12",df_dParms,fitXMin,fitXMax,1);
   derivMS_par12->SetParameter(0,12.);
   TF1 * derivMS_par13 = new TF1("dfdp13",df_dParms,fitXMin,fitXMax,1);
   derivMS_par13->SetParameter(0,13.);
   TF1 * derivMS_par14 = new TF1("dfdp14",df_dParms,fitXMin,fitXMax,1);
   derivMS_par14->SetParameter(0,14.);
   TF1 * derivMS_par15 = new TF1("dfdp15",df_dParms,fitXMin,fitXMax,1);
   derivMS_par15->SetParameter(0,15.);
   TF1 * derivMS_par16 = new TF1("dfdp16",df_dParms,fitXMin,fitXMax,1);
   derivMS_par16->SetParameter(0,16.);
   TF1 * derivMS_par17 = new TF1("dfdp17",df_dParms,fitXMin,fitXMax,1);
   derivMS_par17->SetParameter(0,17.);
   TF1 * derivMS_par18 = new TF1("dfdp18",df_dParms,fitXMin,fitXMax,1);
   derivMS_par18->SetParameter(0,18.);
   TF1 * derivMS_par19 = new TF1("dfdp19",df_dParms,fitXMin,fitXMax,1);
   derivMS_par19->SetParameter(0,19.);
   TF1 * derivMS_par20 = new TF1("dfdp20",df_dParms,fitXMin,fitXMax,1);
   derivMS_par20->SetParameter(0,20.);

   coe[0] = derivMS_par0->Integral(fitXMin,fitXMax); 
   coe[1] = derivMS_par1->Integral(fitXMin,fitXMax); 
   coe[2] = derivMS_par2->Integral(fitXMin,fitXMax); 
   coe[3] = derivMS_par3->Integral(fitXMin,fitXMax); 
   coe[4] = derivMS_par4->Integral(fitXMin,fitXMax); 
   coe[5] = derivMS_par5->Integral(fitXMin,fitXMax); 
   coe[6] = derivMS_par6->Integral(fitXMin,fitXMax); 
   coe[7] = derivMS_par7->Integral(fitXMin,fitXMax); 
   coe[8] = derivMS_par8->Integral(fitXMin,fitXMax); 
   coe[9] = derivMS_par9->Integral(fitXMin,fitXMax); 
   coe[10] = derivMS_par10->Integral(fitXMin,fitXMax); 
   coe[11] = derivMS_par11->Integral(fitXMin,fitXMax); 
   coe[12] = derivMS_par12->Integral(fitXMin,fitXMax); 
   coe[13] = derivMS_par13->Integral(fitXMin,fitXMax); 
   coe[14] = derivMS_par14->Integral(fitXMin,fitXMax); 
   coe[15] = derivMS_par15->Integral(fitXMin,fitXMax); 
   coe[16] = derivMS_par16->Integral(fitXMin,fitXMax); 
   coe[17] = derivMS_par17->Integral(fitXMin,fitXMax); 
   coe[18] = derivMS_par18->Integral(fitXMin,fitXMax); 
   coe[19] = derivMS_par19->Integral(fitXMin,fitXMax); 
   coe[20] = derivMS_par20->Integral(fitXMin,fitXMax); 

   double erri2MS = 0;

   for(int i = 0; i<nPars; i++){
     double p, dp;
     minuit.GetParameter( i, p, dp );
     erri2MS += coe[i] * coe[i] * dp * dp;
   }

   cout << sqrt(erri2MS)/4.0 << endl;
   //   cout << "ERROR_Nms = " << sqrt(erri2MS)/4.0 << endl;

}


int main() {
  //  gROOT->Reset();
  zMinuit3Independentfit_conv();
  return 0;
}

