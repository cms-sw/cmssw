#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuMuScaledFunction.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuTrackScaledFunction.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuStandaloneScaledFunction.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuTrackScaledNormalBack.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuMuNormalBack.h"
//#include "PhysicsTools/Utilities/interface/HistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/MultiHistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootFunctionAdapter.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "TMinuit.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TF1.h"
#include "TMath.h"
#include "TStyle.h"
#include <iostream>
#include <cmath>
#include <string>
#include <boost/shared_ptr.hpp>

using namespace std;

static const int fitXMin = 80; 
static const int fitXMax = 104;
// assuming binning : Zmm = 200; Zmt = 100; Zms = 50; than 
static const int BinScaleFactorZmm = 1;
static const int BinScaleFactorZmt = 2;
static const int BinScaleFactorZms = 4;
const size_t nPars = 17;

static size_t fullBins = 0;

TF1 * fitZmm;
TF1 * fitZmt;  
TF1 * fitZms;
TF1 * bkgZmt;
TF1 * fitZmt_sig;

const size_t nBins = 200; 
const size_t n = nBins, nn = nBins/BinScaleFactorZmt, nnn = nBins/BinScaleFactorZms;

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


template<typename F, typename Minuit>
double error(const Minuit &rMinuit, F& f, double min, double max, size_t pars) {
  double coe[1024];
  for(size_t i = 0;i < pars; ++i)
    coe[i] = root::gradient(f, i, min, max, pars);
  double err = 0;
  for(size_t i = 0; i < nPars; ++i){
    double p, dp;
    p = rMinuit.getParameter(i, dp);
    dp = rMinuit.getParameterError(i, p);
    err += coe[i]*coe[i]*dp*dp;
  }
  return err;
}
  

void zMinuit3Independentfit_conv() {
  cout << ">>> starting fit program" << endl;
  string names[] = { "P0", "mZmm", "GZmm", 
		     "f_gamma", "f_int", 
		     "eff_track","eff_standalone", 
		     "sigmaZmm", "meanZmm","sigmaZmt", "meanZmt", "sigmaZms", "meanZms", 
		     "f_backZmt", "l_backZmt", "a_backZmt", "b_backZmt" 
		     //, "f_backZmm", "l_backZmm", "a_backZmm", "b_backZmm"
		     //, "f_backZms", "l_backZms", "a_backZms", "b_backZms"
  };
  
  
  cout << ">>> set fit parms" << endl;
  using namespace boost;
  shared_ptr<double> 
    P0(new double(6000)), 
    mZmm(new double(91.3)), 
    GZmm(new double(2.5)), 
    f_gamma(new double(0)), 
    f_int(new double(0.001)),  
    eff_track(new double(0.98)), 
    eff_standalone(new double(0.97)), 
    sigmaZmm(new double(1.)), 
    meanZmm(new double(0)), 
    sigmaZmt(new double(1.)), 
    meanZmt(new double(1.)), 
    sigmaZms(new double(5.)), 
    meanZms(new double(0)), 
    f_backZmt(new double(400.)), 
    l_backZmt(new double(0.01)), 
    a_backZmt(new double(0)), 
    b_backZmt(new double(0));
  //f_backZmm(new double(0)), l_backZmm(new double(0.01)), a_backZmm(new double(100)), b_backZmm(new double(0)),
  //f_backZms(new double(50.)), l_backZms(new double(0.0001)), a_backZms(new double(5.)), b_backZms(new double(0.1));

  funct::ZMuMuScaledFunction zmms(mZmm, GZmm, f_gamma, f_int, meanZmm, sigmaZmm, P0, eff_track, eff_standalone, BinScaleFactorZmm);
  funct::ZMuTrackScaledNormalBack zmtsnb(mZmm, GZmm, f_gamma, f_int, meanZmt, sigmaZmt, P0, eff_track, eff_standalone, f_backZmt, l_backZmt, a_backZmt, b_backZmt, BinScaleFactorZmt, fitXMin, fitXMax);
  funct::ZMuStandaloneScaledFunction zmss(mZmm, GZmm, f_gamma, f_int, meanZms, sigmaZms, P0, eff_track, eff_standalone, BinScaleFactorZms);
  funct::ZMuMuNormalBack zmmnb(P0, mZmm, GZmm, f_gamma, fitXMin, fitXMax);
  funct::ZMuTrackScaledFunction zmts(mZmm, GZmm, f_gamma, f_int, meanZmt, sigmaZmt, P0, eff_track, eff_standalone, BinScaleFactorZmt);
  
  TFile * ZToLL_file1 = new TFile("../test/ZMM_ZMT_ZMS_histo/ZCandidates_Histo_iso_all.root","read");
  TH1D * zToMuMu = (TH1D*) ZToLL_file1->Get("zToMM");
  TH1D * zToSingleTrackMu = (TH1D*) ZToLL_file1->Get("zToMTk");
  
  TH1D * zToSingleStandAloneMu = (TH1D*) ZToLL_file1->Get("zToMS");

  typedef MultiHistoChiSquare<ZMuMuScaledFunction, ZMuTrackScaledNormalBack, ZMuStandaloneScaledFunction> ChiSquared;
  ChiSquared chiZMM(zmms, zToMuMu, BinScaleFactorZmm, 
		    zmtsnb, zToSingleTrackMu, BinScaleFactorZmt, 
		    zmss, zToSingleStandAloneMu, BinScaleFactorZms,
		    fitXMin, fitXMax);
  fullBins = chiZMM.degreesOfFreedom();
  cout << "fullBins === " << fullBins << endl;
  
  cout << ">>> initializing fit" << endl;
  
  fit::RootMinuit<ChiSquared> rMinuit(nPars, chiZMM, true);
  //set parameters
  rMinuit.setParameter(0, names[0].c_str(), P0, 10, 100, 100000);
  rMinuit.setParameter(1, names[1].c_str(), mZmm, .1, 70., 110);
  rMinuit.setParameter(2, names[2].c_str(), GZmm, 1, 1, 10);
  rMinuit.setParameter(3, names[3].c_str(), f_gamma, 0.1, -100, 1000);
  rMinuit.setParameter(4, names[4].c_str(), f_int, .0001, -1000000, 1000000);
  rMinuit.setParameter(5, names[5].c_str(), eff_track, .01, 0.,1.);
  rMinuit.setParameter(6, names[6].c_str(), eff_standalone, .01, 0., 1.);
  rMinuit.setParameter(7, names[7].c_str(), sigmaZmm, .1, 0., 5.);
  rMinuit.setParameter(8, names[8].c_str(), meanZmm, .001, 0., .5);
  rMinuit.setParameter(9, names[9].c_str(), sigmaZmt, .1, 0., 3.);
  rMinuit.setParameter(10, names[10].c_str(), meanZmt, .1, 0., 10.);
  rMinuit.setParameter(11, names[11].c_str(), sigmaZms, .1, 0., 15.);
  rMinuit.setParameter(12, names[12].c_str(), meanZms, .1, 0., 1.);
  rMinuit.setParameter(13, names[13].c_str(), f_backZmt, 1, 0, 10000);
  rMinuit.setParameter(14, names[14].c_str(), l_backZmt,  .001, 0, 100);
  rMinuit.setParameter(15, names[15].c_str(), a_backZmt, 1, -10000, 10000);
  rMinuit.setParameter(16, names[16].c_str(), b_backZmt, 1, -1000, 1000);
  rMinuit.fixParameter(3);
  rMinuit.fixParameter(8);
  rMinuit.fixParameter(15);
  rMinuit.fixParameter(16);
  /*
  rMinuit.setParameter(17, names[17].c_str(), f_backZmm, .1, 0, 60);
  rMinuit.setParameter(18, names[18].c_str(), l_backZmm,  0.001,0., 0.1);
  rMinuit.setParameter(19, names[19].c_str(), a_backZmm, .1, 0, 10000); 
  rMinuit.setParameter(20, names[20].c_str(), b_backZmm, .0001, -1000, 1000);
  rMinuit.setParameter(21, names[21].c_str(), f_backZms, 1, -10, 100000);
  rMinuit.setParameter(22, names[22].c_str(), l_backZms,  .0001, -100, 100);
  rMinuit.setParameter(23, names[23].c_str(), a_backZms, 1, -10000, 10000);
  rMinuit.setParameter(24, names[24].c_str(), b_backZms, 1, -1000, 1000);
  rMinuit.fixParameter(17);
  rMinuit.fixParameter(20);
  */

  cout << ">>> starting fit" << endl;
  double amin = rMinuit.minimize();
  
   cout << "fullBins = " << fullBins << "; free pars = " << rMinuit.getNumberOfFreeParameters() << endl;
  unsigned int ndof = fullBins - rMinuit.getNumberOfFreeParameters();
  cout << "Chi^2 = " << amin << "/" << ndof << " = " << amin/ndof 
       << "; prob: " << TMath::Prob( amin, ndof )
       << endl;
  int ZMMPars = ZMuMuScaledfunction::parameters;
  int ZMTBPars = ZMuTrackScaledNormalBack::parameters;
  int ZMSPars = ZMuStandaloneScaledfunction::parameters;
  int ZMMBPars = ZMuMuNormalBack::parameters;
  int ZMTPars = ZMuTrackScaledfunction::parameters;
  double p[nPars], dp[nPars];
  for( size_t i = 0; i < nPars; ++ i ) {
    p[i] = rMinuit.getParameter(i, dp[i]);
    dp[i] = rMinuit.getParameterError(i, p[i]);
    cout << names[i] << " = " << p[i] << "+/-" << dp[i] << endl;
  }
  double numberOfEvents = p[0];
  double ZmmMass = p[1]; 
  double ZmmWidth = p[2];
  double gammaFactor = p[3]; 
  double interferenceFactor = p[4];  
  double trackEfficiency = p[5]; 
  double standaloneEfficiency = p[6]; 
  double ZmmSigma = p[7];  
  double ZmmMean = p[8]; 
  double ZmtSigma = p[9]; 
  double ZmtMean = p[10];
  double ZmsSigma =  p[11]; 
  double ZmsMean = p[12]; 
  double ZmtBkgN = p[13]; 
  double ZmtBkgLambda = p[14]; 
  double ZmtBkgA1 = p[15]; 
  double ZmtBkgA2 = p[16];

  //   TF1 * fitZmm = new TF1("fitZmm", root::function(zmms), fitXMin, fitXMax, ZMMPars); 
  //   TF1 * fitZmt = new TF1("fitZmt", root::function(zmtsnb), fitXMin, fitXMax, ZMTBPars);
  //   TF1 * fitZms = new TF1("fitZms", root::function(zmss), fitXMin, fitXMax, ZMSPars); 
  fitZmm = new TF1("fitZmm", root::function(zmms), fitXMin, fitXMax, ZMMPars); 
  double fitZmmPars[] = { ZmmMass, ZmmWidth, gammaFactor, interferenceFactor, 
			  ZmmMean, ZmmSigma, 
			  numberOfEvents, trackEfficiency, standaloneEfficiency };
  fitZmm->SetParameters(fitZmmPars);
  fitZmm->SetParNames("ZmmMass","ZmmWidth","gammaFactor","interferenceFactor",
                      "ZmmMean","ZmmSigma",
                      "numberOfEvents","trackEfficiency","standaloneEfficiency");
  fitZmm->SetLineColor(kRed);
  fitZmt = new TF1("fitZmt", root::function(zmtsnb), fitXMin, fitXMax, ZMTBPars);
  double fitZmtPars[] = {ZmmMass, ZmmWidth, gammaFactor, interferenceFactor, 
			 ZmtMean, ZmtSigma, 
			 numberOfEvents, trackEfficiency, standaloneEfficiency, 
			 ZmtBkgN, ZmtBkgLambda, ZmtBkgA1, ZmtBkgA2};
  fitZmt->SetParameters(fitZmtPars);
  fitZmt->SetParNames("ZmmMass","ZmmWidth","gammaFactor","interferenceFactor",
                      "ZmtMean","ZmtSigma",
                      "numberOfEvents","trackEfficiency","standaloneEfficiency");
  fitZmt->SetParName(9, "ZmtBkgN"); //max number of parameters for ROOT: 11!
  fitZmt->SetParName(10, "ZmtBkgLambda");
  fitZmt->SetParName(11, "ZmtBkgA1");
  fitZmt->SetParName(12, "ZmtBkgA2");
  fitZmt->SetLineColor(kRed);
  fitZms = new TF1("fitZms", root::function(zmss), fitXMin, fitXMax, ZMSPars);
  double fitZmsPars[] = {ZmmMass, ZmmWidth, gammaFactor, interferenceFactor, 
			 ZmsMean, ZmsSigma, 
			 numberOfEvents, trackEfficiency, standaloneEfficiency};
  fitZms->SetParameters(fitZmsPars);
  fitZms->SetParNames("ZmmMass","ZmmWidth","gammaFactor","interferenceFactor",
                      "ZmsMean","ZmsSigma",
                      "numberOfEvents","trackEfficiency","standaloneEfficiency");
  fitZms->SetLineColor(kRed);
  //TF1 * bkgZmm = new TF1("bkgZmm",root::function(zmmnb), fitXMin, fitXMax, ZMMBPars);
  //TF1 * bkgZmt = new TF1("bkgZmt",root::function(zmmnb), fitXMin, fitXMax, ZMMBPars);
  //TF1 * bkgZms = new TF1("bkgZms",root::function(zmmnb), fitXMin, fitXMax, ZMMBPars);
  bkgZmt = new TF1("bkgZmt", root::function(zmmnb), fitXMin, fitXMax, ZMMBPars);
  double bkgZmtPars[] = {ZmtBkgN, ZmtBkgLambda, ZmtBkgA1, ZmtBkgA2};
  bkgZmt->SetParameters(bkgZmtPars);
  bkgZmt->SetParNames("ZmtBkgN", "ZmtBkgLambda","ZmtBkgA1","ZmtBkgA2");
  bkgZmt->SetLineColor(kGreen);
  // Signal-only function in the Zmt case
  fitZmt_sig = new TF1("fitZmt_sig", root::function(zmts), fitXMin, fitXMax, ZMTPars); 
  double fitZmt_sigPars[] = {ZmmMass, ZmmWidth, gammaFactor, interferenceFactor, 
			     ZmtMean, ZmtSigma, 
			     numberOfEvents, trackEfficiency, standaloneEfficiency};
  fitZmt_sig->SetParameters(fitZmt_sigPars);
  fitZmt_sig->SetParNames("ZmmMass","ZmmWidth","gammaFactor","interferenceFactor",
                          "ZmtMean","ZmtSigma",
                          "numberOfEvents","trackEfficiency","standaloneEfficiency"); 
  
  gStyle->SetOptFit(1111);
  // gStyle->SetOptLogy();
  
  /*for( size_t i = 0; i < nPars; ++ i ) {
    double p, dp;
    minuit.GetParameter( i, p, dp );
    cout << names[i] << " = " << p << "+/-" << dp << endl;
  }*/
  /*
  for( size_t i = 0; i < nPars; ++ i ) {
    double p, dp;
    p = rMinuit.getParameter(i, dp);
    dp = rMinuit.getParameterError(i, p);
    cout << names[i] << " = " << p << "+/-" << dp << endl;
  }
  */

  /*
  for(size_t i = 0; i < nPars; ++i) { 
    double p, dp;
    p = rMinuit.getParameter(i, dp);
    dp = rMinuit.getParameterError(i, p);
    fitZmm->SetParameter(i,p);
    fitZmt->SetParameter(i,p);
    fitZms->SetParameter(i,p);
    if(i < 4) bkgZmt->SetParameter(i,p);
    if(i < 13) fitZmt_sig->SetParameter(i,p);
  }
  */

  gStyle->SetOptLogy();  
  TCanvas c1;
  
  zToMuMu->Draw("e");
  //zToMuMu->GetXaxis()->SetRangeUser(85,97);
  
  zToMuMu->GetXaxis()->SetRangeUser(fitXMin-10, fitXMax+10);
  zToMuMu->SetXTitle("#mu #mu mass (GeV/c^{2})");
  zToMuMu->SetTitle("fit of #mu #mu mass with isolation cut");
  zToMuMu->SetYTitle("Events / 0.1 GeV/c^{2}");
  
  
  fitZmm->Draw("same");
  //bkgZmm->Draw("same");
  c1.SaveAs("zmm.eps");
  
  TCanvas c2;
  
  zToSingleTrackMu->Draw("e");
  
  zToSingleTrackMu->GetXaxis()->SetRangeUser(fitXMin-20, fitXMax+20);
  zToSingleTrackMu->SetXTitle("#mu + (unmatched) track mass (GeV/c^{2})");
  zToSingleTrackMu->SetTitle("fit of #mu + (unmatched) track  mass with isolation cut");
  zToSingleTrackMu->SetYTitle("Events / 2 GeV/c^{2}");
  //  bkgZmt->Draw("same");
  fitZmt->Draw("same");
  bkgZmt->Draw("same");
  c2.SaveAs( "zmt.eps");
  
  TCanvas c3;
  
  zToSingleStandAloneMu->Draw("e");
  zToSingleStandAloneMu->GetXaxis()->SetRangeUser(fitXMin-20, fitXMax+20);
  zToSingleStandAloneMu->SetXTitle("#mu + (unmatched) standalone mass (GeV/c^{2})");
  zToSingleStandAloneMu->SetTitle("fit of #mu + (unmatched) standalone  mass with isolation cut");
  zToSingleStandAloneMu->SetYTitle("Events / 4 GeV/c^{2}");
  //bkgZms->Draw("same");
  fitZms->Draw("same");
  c3.SaveAs( "zms.eps");
  
  //   fitZmm->SetParameter(4,0);
  //double p4, dp4;
  //minuit.GetParameter( 4, p4, dp4 );
  //   cout <<"Integral of Zmm == "<<fitZmm->Integral(fitXMin ,fitXMax)<<endl;
  double Int = fitZmm->Integral(fitXMin,fitXMax);
  int scale =  (int) (1./BinScaleFactorZmm);
  cout <<"scale == "<<scale<<endl;
  double IntHisto = zToMuMu->Integral((scale *fitXMin)+1, (scale* fitXMax)+1);
  
  cout <<"Integral of Zmm == "<<Int<<" vs Histo Integral "<<IntHisto<<endl;   
  /* 
     double p0, dp0;
     minuit.GetParameter( 0, p0, dp0 );
     double p5, dp5;
     minuit.GetParameter( 5, p5, dp5 );
     doubfitZmt_sigParsle p6, dp6;
     minuit.GetParameter( 6, p6, dp6 );
     double p17, dp17;
     minuit.GetParameter( 17, p17, dp17 );
  */
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
  //  TF1 * derivMM_par0 = new TF1("dfdp0",df_dParmm,fitXMin,fitXMax,1);
  //  dearivMM_par0->SetParameter(0,0);


  double erri2MM = error(rMinuit, zmms, fitXMin,fitXMax, nPars);
  cout << sqrt(erri2MM) << endl;

  double coe[1024];
  
  
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
  
  for(size_t i = 0; i < 13; ++i){
    double p, dp;
    p = rMinuit.getParameter(i, dp);
    dp = rMinuit.getParameterError(i, p);
    //minuit.GetParameter( i, p, dp );
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
  
  for(size_t i = 0; i < nPars; ++i){
    double p, dp;
    p = rMinuit.getParameter(i, dp);
    dp = rMinuit.getParameterError(i, p);
    //minuit.GetParameter( i, p, dp );
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

