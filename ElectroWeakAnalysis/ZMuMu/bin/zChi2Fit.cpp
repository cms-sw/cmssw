#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include "PhysicsTools/Utilities/interface/Numerical.h"
#include "PhysicsTools/Utilities/interface/Exponential.h"
#include "PhysicsTools/Utilities/interface/Polynomial.h"
#include "PhysicsTools/Utilities/interface/Constant.h"
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/MultiHistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/HistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/HistoPoissonLikelihoodRatio.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/RootMinuitCommands.h"
#include "PhysicsTools/Utilities/interface/FunctClone.h"
#include "PhysicsTools/Utilities/interface/rootPlot.h"
#include "PhysicsTools/Utilities/interface/Expression.h"
#include "PhysicsTools/Utilities/interface/HistoPdf.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TH1.h"
#include "TFile.h"
#include <fstream>
#include <iostream>
#include <algorithm> 
#include <exception>
#include <iterator>
#include <string>
#include <vector>
using namespace std;

// A helper function to simplify the main part.
template<class T>
ostream& operator<<(ostream& os, const vector<T>& v) {
  copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
  return os;
}

// A function that get histogram and sets contents to 0 
// if entries are too small
TH1 * getHisto(TFile * file, const char * name, unsigned int rebin) {
  TObject * h = file->Get(name);
  if(h == 0)
    throw edm::Exception(edm::errors::Configuration) 
      << "Can't find object " << name << "\n";
  TH1 * histo = dynamic_cast<TH1*>(h);
  if(histo == 0)
    throw edm::Exception(edm::errors::Configuration) 
      << "Object " << name << " is of type " << h->ClassName() << ", not TH1\n";
  histo->Rebin(rebin);  
  for(int i = 1; i <= histo->GetNbinsX(); ++i) {
    if(histo->GetBinContent(i) < 0.1) {
      histo->SetBinContent(i, 0.0);
      histo->SetBinError(i, 0.0);
    }
  }
  return histo;
}

struct sig_tag;
struct bkg_tag;

typedef funct::FunctExpression Expr;
typedef fit::HistoChiSquare<funct::FunctExpression> ExprChi2;
typedef fit::HistoPoissonLikelihoodRatio<funct::FunctExpression> ExprPLR;

double fMin, fMax;
 unsigned int rebinMuMuNoIso ,rebinMuMu =1 , rebinMuMu1HLT , rebinMuMu2HLT , rebinMuTk , rebinMuSa;
 // assume that the bin size is 1 GeV!!! 
string ext, region;
bool nonIsoTemplateFromMC;

template<typename T>
struct PlotPrefix { };

template<>
struct PlotPrefix<ExprChi2> {
  static string str() { return "chi2"; }
};
 
template<>
struct PlotPrefix<ExprPLR> {
  static string str() { return "plr"; }
};
 
template<typename T>
int main_t(const vector<string> & v_file){
  typedef fit::MultiHistoChiSquare<T, T, T, T, T> ChiSquared;
  fit::RootMinuitCommands<ChiSquared> commands("zChi2Fit.txt");
  
  cout << "minuit command file completed" << endl;
 
  funct::Constant rebinMuMuNoIsoConst(rebinMuMuNoIso), rebinMuMuConst(rebinMuMu), 
    rebinMuMu1HLTConst(rebinMuMu1HLT), rebinMuMu2HLTConst(rebinMuMu2HLT), 
    rebinMuTkConst(rebinMuTk), rebinMuSaConst(rebinMuSa);
  
  for(vector<string>::const_iterator it = v_file.begin(); it != v_file.end(); ++it) {
    TFile * root_file = new TFile(it->c_str(), "read");
    
    // default when region==all
    //    TH1 * histoZMuMuNoIso = getHisto(root_file, "nonIsolatedZToMuMuPlots/zMass",rebinMuMuNoIso);
    TH1 * histoZMuMuNoIso = getHisto(root_file, "oneNonIsolatedZToMuMuPlots/zMass",rebinMuMuNoIso);
    TH1 * histoZMuMu = getHisto(root_file, "goodZToMuMuPlots/zMass",rebinMuMu);
    TH1 * histoZMuMu1HLT = getHisto(root_file, "goodZToMuMu1HLTPlots/zMass", rebinMuMu1HLT);
    TH1 * histoZMuMu2HLT = getHisto(root_file, "goodZToMuMu2HLTPlots/zMass", rebinMuMu2HLT);
    TH1 * histoZMuTk = getHisto(root_file, "goodZToMuMuOneTrackPlots/zMass", rebinMuTk);
    TH1 * histoZMuSa = getHisto(root_file, "goodZToMuMuOneStandAloneMuonPlots/zMass", rebinMuSa);
    TH1 * histoZMuSaFromMuMu = getHisto(root_file, "zmumuSaMassHistogram/zMass", rebinMuSa);
   
    TH1 * histoZMuMuNoIsoTemplateFromMC= histoZMuMu;
    if (nonIsoTemplateFromMC) {
      //      histoZMuMuNoIsoTemplateFromMC = getHisto(root_file, "nonIsolatedZToMuMuPlotsMC/zMass",rebinMuMu);
      histoZMuMuNoIsoTemplateFromMC = getHisto(root_file, "oneNonIsolatedZToMuMuPlotsMC/zMass",rebinMuMu);
  }    
    if (region=="barrel"){
      histoZMuMuNoIso = getHisto(root_file, "nonIsolatedZToMuMuPlotsBarrel/zMass",rebinMuMuNoIso);
      histoZMuMu = getHisto(root_file, "goodZToMuMuPlotsBarrel/zMass",rebinMuMu);
      histoZMuMu1HLT = getHisto(root_file, "goodZToMuMu1HLTPlotsBarrel/zMass", rebinMuMu1HLT);
      histoZMuMu2HLT = getHisto(root_file, "goodZToMuMu2HLTPlotsBarrel/zMass", rebinMuMu2HLT);
      histoZMuTk = getHisto(root_file, "goodZToMuMuOneTrackPlotsBarrel/zMass", rebinMuTk);
      histoZMuSa = getHisto(root_file, "goodZToMuMuOneStandAloneMuonPlotsBarrel/zMass", rebinMuSa);
      histoZMuSaFromMuMu = getHisto(root_file, "zmumuSaMassHistogramBarrel/zMass", rebinMuSa);
    }
    
    if (region=="endcap"){
      histoZMuMuNoIso = getHisto(root_file, "nonIsolatedZToMuMuPlotsEndCap/zMass",rebinMuMuNoIso);
      histoZMuMu = getHisto(root_file, "goodZToMuMuPlotsEndCap/zMass",rebinMuMu);
      histoZMuMu1HLT = getHisto(root_file, "goodZToMuMu1HLTPlotsEndCap/zMass", rebinMuMu1HLT);
      histoZMuMu2HLT = getHisto(root_file, "goodZToMuMu2HLTPlotsEndCap/zMass", rebinMuMu2HLT);
      histoZMuTk = getHisto(root_file, "goodZToMuMuOneTrackPlotsEndCap/zMass", rebinMuTk);
      histoZMuSa = getHisto(root_file, "goodZToMuMuOneStandAloneMuonPlotsEndCap/zMass", rebinMuSa);
      histoZMuSaFromMuMu = getHisto(root_file, "zmumuSaMassHistogramEndCap/zMass", rebinMuSa);
    }
    
    if (region=="barrend"){
      histoZMuMuNoIso = getHisto(root_file, "nonIsolatedZToMuMuPlotsBarrEnd/zMass",rebinMuMuNoIso);
      histoZMuMu = getHisto(root_file, "goodZToMuMuPlotsBarrEnd/zMass",rebinMuMu);
      histoZMuMu1HLT = getHisto(root_file, "goodZToMuMu1HLTPlotsBarrEnd/zMass", rebinMuMu1HLT);
      histoZMuMu2HLT = getHisto(root_file, "goodZToMuMu2HLTPlotsBarrEnd/zMass", rebinMuMu2HLT);
      histoZMuTk = getHisto(root_file, "goodZToMuMuOneTrackPlotsBarrEnd/zMass", rebinMuTk);
      histoZMuSa = getHisto(root_file, "goodZToMuMuOneStandAloneMuonPlotsBarrEnd/zMass", rebinMuSa);
      histoZMuSaFromMuMu = getHisto(root_file, "zmumuSaMassHistogramBarrEnd/zMass", rebinMuSa);
    }
    
    if (region!="endcap" && region!="barrel" && region!="barrend" && region!="all"  ){
      cout<< "not a valid region selected"<< endl;
      cout << "possible choises are: all, barrel, endcap, barrend " << endl;
      return 0;
    }
    
    cout << ">>> histogram loaded\n";
    string f_string = *it + "_" + PlotPrefix<T>::str() + "_";
    replace(f_string.begin(), f_string.end(), '.', '_');
    replace(f_string.begin(), f_string.end(), '/', '_');
    string plot_string = f_string + "." + ext;
    cout << ">>> Input files loaded\n";
    
    const char * kYieldZMuMu = "YieldZMuMu";
    const char * kEfficiencyTk = "EfficiencyTk";
    const char * kEfficiencySa = "EfficiencySa";
    const char * kEfficiencyIso = "EfficiencyIso";
    const char * kEfficiencyHLT = "EfficiencyHLT"; 
    const char * kYieldBkgZMuTk = "YieldBkgZMuTk"; 
    const char * kYieldBkgZMuSa = "YieldBkgZMuSa"; 
    const char * kYieldBkgZMuMuNotIso = "YieldBkgZMuMuNotIso";
    const char * kAlpha = "Alpha";
    const char * kBeta = "Beta";
    const char * kLambda = "Lambda";
    const char * kA0 = "A0"; 
    const char * kA1 = "A1"; 
    const char * kA2 = "A2"; 
    const char * kB0 = "B0"; 
    const char * kB1 = "B1"; 
    const char * kB2 = "B2"; 
    const char * kC0 = "C0"; 
    const char * kC1 = "C1"; 
    const char * kC2 = "C2"; 
    
    funct::Parameter yieldZMuMu(kYieldZMuMu, commands.par(kYieldZMuMu));
    funct::Parameter effTk(kEfficiencyTk, commands.par(kEfficiencyTk)); 
    funct::Parameter effSa(kEfficiencySa, commands.par(kEfficiencySa)); 
    funct::Parameter effIso(kEfficiencyIso, commands.par(kEfficiencyIso)); 
    funct::Parameter effHLT(kEfficiencyHLT, commands.par(kEfficiencyHLT)); 
    funct::Parameter yieldBkgZMuTk(kYieldBkgZMuTk, commands.par(kYieldBkgZMuTk));
    funct::Parameter yieldBkgZMuSa(kYieldBkgZMuSa, commands.par(kYieldBkgZMuSa));
    funct::Parameter yieldBkgZMuMuNotIso(kYieldBkgZMuMuNotIso, commands.par(kYieldBkgZMuMuNotIso));
    funct::Parameter alpha(kAlpha, commands.par(kAlpha));
    funct::Parameter beta(kBeta, commands.par(kBeta));
    funct::Parameter lambda(kLambda, commands.par(kLambda));
    funct::Parameter a0(kA0, commands.par(kA0));
    funct::Parameter a1(kA1, commands.par(kA1));
    funct::Parameter a2(kA2, commands.par(kA2));
    funct::Parameter b0(kB0, commands.par(kB0));
    funct::Parameter b1(kB1, commands.par(kB1));
    funct::Parameter b2(kB2, commands.par(kB2));
    funct::Parameter c0(kC0, commands.par(kC0));
    funct::Parameter c1(kC1, commands.par(kC1));
    funct::Parameter c2(kC2, commands.par(kC2));
    funct::Constant cFMin(fMin), cFMax(fMax);
    
    // count ZMuMu Yield
    double nZMuMu = 0, nZMuMu1HLT = 0, nZMuMu2HLT = 0;
    {
      unsigned int nBins = histoZMuMu->GetNbinsX();
      double xMin = histoZMuMu->GetXaxis()->GetXmin();
      double xMax = histoZMuMu->GetXaxis()->GetXmax();
      double deltaX =(xMax - xMin) / nBins;
      for(unsigned int i = 0; i < nBins; ++i) { 
	double x = xMin + (i +.5) * deltaX;
	if(x > fMin && x < fMax){
	  nZMuMu += histoZMuMu->GetBinContent(i+1);
	  nZMuMu1HLT += histoZMuMu1HLT->GetBinContent(i+1);
	  nZMuMu2HLT += histoZMuMu2HLT->GetBinContent(i+1);
	}
      }
    }
    // aggiungi 1HLT 2HLT
    cout << ">>> count of ZMuMu yield in the range [" << fMin << ", " << fMax << "]: " << nZMuMu << endl;
    cout << ">>> count of ZMuMu (1HLT) yield in the range [" << fMin << ", " << fMax << "]: "  << nZMuMu1HLT << endl;
    cout << ">>> count of ZMuMu (2HLT) yield in the range [" << fMin << ", " << fMax << "]: "  << nZMuMu2HLT << endl;
    funct::RootHistoPdf zPdfMuMu(*histoZMuMu, fMin, fMax);
      //assign ZMuMu as pdf
    funct::RootHistoPdf zPdfMuMuNonIso = zPdfMuMu;
    if (nonIsoTemplateFromMC) {
      funct::RootHistoPdf zPdfMuMuNoIsoFromMC(*histoZMuMuNoIsoTemplateFromMC, fMin, fMax);
      zPdfMuMuNonIso = zPdfMuMuNoIsoFromMC;
    }    
    
    funct::RootHistoPdf zPdfMuTk = zPdfMuMu;
    funct::RootHistoPdf zPdfMuMu1HLT = zPdfMuMu;
    funct::RootHistoPdf zPdfMuMu2HLT = zPdfMuMu;
    funct::RootHistoPdf zPdfMuSa(*histoZMuSaFromMuMu, fMin, fMax);
    zPdfMuMuNonIso.rebin(rebinMuMuNoIso/rebinMuMu);
    zPdfMuTk.rebin(rebinMuTk/rebinMuMu);
    zPdfMuMu1HLT.rebin(rebinMuMu1HLT/rebinMuMu);
    zPdfMuMu2HLT.rebin(rebinMuMu2HLT/rebinMuMu);
    
    funct::Numerical<2> _2;
    funct::Numerical<1> _1;
    
    //Efficiency term
    Expr zMuMuEff1HLTTerm = _2 * (effTk ^ _2) *  (effSa ^ _2) * (effIso ^ _2) * effHLT * (_1 - effHLT); 
    Expr zMuMuEff2HLTTerm = (effTk ^ _2) *  (effSa ^ _2) * (effIso ^ _2) * (effHLT ^ _2) ; 
    //    Expr zMuMuNoIsoEffTerm = (effTk ^ _2) * (effSa ^ _2) * (_1 - (effIso ^ _2)) * (_1 - ((_1 - effHLT)^_2));
    // change to both hlt and one not iso
    Expr zMuMuNoIsoEffTerm = _2 * (effTk ^ _2) * (effSa ^ _2) * effIso * (_1 - effIso) * (effHLT^_2);
    Expr zMuTkEffTerm = _2 * (effTk ^ _2) * effSa * (_1 - effSa) * (effIso ^ _2) * effHLT;
    Expr zMuSaEffTerm = _2 * (effSa ^ _2) * effTk * (_1 - effTk) * (effIso ^ _2) * effHLT;
    
    Expr zMuMu1HLT = rebinMuMu1HLTConst * zMuMuEff1HLTTerm * yieldZMuMu;
    Expr zMuMu2HLT = rebinMuMu2HLTConst * zMuMuEff2HLTTerm * yieldZMuMu;
    
    Expr zMuTkBkg = yieldBkgZMuTk * funct::Exponential(lambda)* funct::Polynomial<2>(a0, a1, a2);
    Expr zMuTkBkgScaled = rebinMuTkConst * zMuTkBkg;
    Expr zMuTk = rebinMuTkConst * (zMuTkEffTerm * yieldZMuMu * zPdfMuTk + zMuTkBkg);
    
    Expr zMuMuNoIsoBkg = yieldBkgZMuMuNotIso * funct::Exponential(alpha)* funct::Polynomial<2>(b0, b1, b2);
    Expr zMuMuNoIsoBkgScaled = rebinMuMuNoIsoConst * zMuMuNoIsoBkg;
    Expr zMuMuNoIso = rebinMuMuNoIsoConst * (zMuMuNoIsoEffTerm * yieldZMuMu * zPdfMuMuNonIso + zMuMuNoIsoBkg);
    

    Expr zMuSaBkg = yieldBkgZMuSa * funct::Exponential(beta)* funct::Polynomial<2>(c0, c1, c2);
    Expr zMuSaBkgScaled = rebinMuSaConst * zMuSaBkg;
    Expr zMuSa = rebinMuSaConst * (zMuSaEffTerm * yieldZMuMu * zPdfMuSa  + zMuSaBkg );
    
    TH1D histoZCount1HLT("histoZCount1HLT", "", 1, fMin, fMax);
    histoZCount1HLT.Fill(100, nZMuMu1HLT);
    TH1D histoZCount2HLT("histoZCount2HLT", "", 1, fMin, fMax);
    histoZCount2HLT.Fill(100, nZMuMu2HLT);
    
    ChiSquared chi2(zMuMu1HLT, & histoZCount1HLT,
		    zMuMu2HLT, & histoZCount2HLT,
		    zMuTk, histoZMuTk, 
		    zMuSa, histoZMuSa, 
		    zMuMuNoIso,histoZMuMuNoIso,
		    fMin, fMax);
    cout << "N. bins: " << chi2.numberOfBins() << endl;
    
    fit::RootMinuit<ChiSquared> minuit(chi2, true);
    commands.add(minuit, yieldZMuMu);
    commands.add(minuit, effTk);
    commands.add(minuit, effSa);
    commands.add(minuit, effIso);
    commands.add(minuit, effHLT);
    commands.add(minuit, yieldBkgZMuTk);
    commands.add(minuit, yieldBkgZMuSa);
    commands.add(minuit, yieldBkgZMuMuNotIso);
    commands.add(minuit, lambda);
    commands.add(minuit, alpha);
    commands.add(minuit, beta);
    commands.add(minuit, a0);
    commands.add(minuit, a1);
    commands.add(minuit, a2);
    commands.add(minuit, b0);
    commands.add(minuit, b1);
    commands.add(minuit, b2);
    commands.add(minuit, c0);
    commands.add(minuit, c1);
    commands.add(minuit, c2);
    commands.run(minuit);
    const unsigned int nPar = 20;//WARNIG: this must be updated manually for now
    ROOT::Math::SMatrix<double, nPar, nPar, ROOT::Math::MatRepSym<double, nPar> > err;
    minuit.getErrorMatrix(err);
    
    std::cout << "error matrix:" << std::endl;
    for(unsigned int i = 0; i < nPar; ++i) {
      for(unsigned int j = 0; j < nPar; ++j) {
	  std::cout << err(i, j) << "\t";
      }
      std::cout << std::endl;
    } 
    minuit.printFitResults();
    ofstream myfile;
    myfile.open ("fitResult.txt", ios::out | ios::app);
    myfile<<"\n";
    double Y =  minuit.getParameterError("YieldZMuMu");
    double dY = minuit.getParameterError("YieldZMuMu", Y);
    double tk_eff =  minuit.getParameterError("EfficiencyTk");
    double dtk_eff = minuit.getParameterError("EfficiencyTk", tk_eff);
    double sa_eff =  minuit.getParameterError("EfficiencySa");
    double dsa_eff = minuit.getParameterError("EfficiencySa", sa_eff);
    double iso_eff =  minuit.getParameterError("EfficiencyIso");
    double diso_eff = minuit.getParameterError("EfficiencyIso", iso_eff);
    double hlt_eff =  minuit.getParameterError("EfficiencyHLT");
    double dhlt_eff = minuit.getParameterError("EfficiencyHLT",hlt_eff);
    myfile<< Y <<" "<< dY <<" "<< tk_eff <<" "<< dtk_eff <<" "<< sa_eff << " " << dsa_eff << " " << iso_eff <<" " << diso_eff<< " " << hlt_eff << " " << dhlt_eff << " " <<chi2()/(chi2.numberOfBins()- minuit.numberOfFreeParameters());
    
    myfile.close();

    //Plot
    double s;
    s = 0;
    for(int i = 1; i <= histoZMuMuNoIso->GetNbinsX(); ++i)
      s += histoZMuMuNoIso->GetBinContent(i);
    histoZMuMuNoIso->SetEntries(s);
    s = 0;
    for(int i = 1; i <= histoZMuMu->GetNbinsX(); ++i)
      s += histoZMuMu->GetBinContent(i);
    histoZMuMu->SetEntries(s);
    s = 0;
    for(int i = 1; i <= histoZMuMu1HLT->GetNbinsX(); ++i)
      s += histoZMuMu1HLT->GetBinContent(i);
    histoZMuMu1HLT->SetEntries(s);
    s = 0;
    for(int i = 1; i <= histoZMuMu2HLT->GetNbinsX(); ++i)
      s += histoZMuMu2HLT->GetBinContent(i);
    histoZMuMu2HLT->SetEntries(s);
    s = 0;
    for(int i = 1; i <= histoZMuTk->GetNbinsX(); ++i)
      s += histoZMuTk->GetBinContent(i);
    histoZMuTk->SetEntries(s);
    s = 0;
    for(int i = 1; i <= histoZMuSa->GetNbinsX(); ++i)
      s += histoZMuSa->GetBinContent(i);
    histoZMuSa->SetEntries(s);
    
    string ZMuMu1HLTPlot = "ZMuMu1HLTFit_" + plot_string;
    root::plot<Expr>(ZMuMu1HLTPlot.c_str(), *histoZMuMu1HLT, zMuMu1HLT, fMin, fMax, 
		     effTk, effSa, effIso, effHLT, yieldZMuMu, 
		     kOrange-2, 2, kSolid, 100, 
		     "Z -> #mu #mu mass", "#mu #mu invariant mass (GeV/c^{2})", 
		     "Events");
    
    string ZMuMu2HLTPlot = "ZMuMu2HLTFit_" + plot_string;
    root::plot<Expr>(ZMuMu2HLTPlot.c_str(), *histoZMuMu2HLT, zMuMu2HLT, fMin, fMax, 
		     effTk, effSa, effIso, effHLT, yieldZMuMu, 
		     kOrange-2, 2, kSolid, 100, 
		     "Z -> #mu #mu mass", "#mu #mu invariant mass (GeV/c^{2})", 
		     "Events");
    
    
    string ZMuMuNoIsoPlot = "ZMuMuNoIsoFit_X_" + plot_string;
    root::plot<Expr>(ZMuMuNoIsoPlot.c_str(), *histoZMuMuNoIso, zMuMuNoIso, fMin, fMax, 
		     effTk, effSa, effIso, effHLT, yieldZMuMu,
		     yieldBkgZMuMuNotIso, alpha, b0, b1, b2,
		     kWhite, 2, kSolid, 100, 
		     "Z -> #mu #mu Not Iso mass", "#mu #mu invariant mass (GeV/c^{2})", 
		     "Events");	
    ZMuMuNoIsoPlot = "ZMuMuNoIsoFit_" + plot_string;
    TF1 funZMuMuNoIso = root::tf1_t<sig_tag, Expr>("ZMuMuNoIsoFunction", zMuMuNoIso, fMin, fMax, 
					      effTk, effSa, effIso, effHLT, yieldZMuMu, 
					      yieldBkgZMuMuNotIso, alpha, b0, b1, b2);
    funZMuMuNoIso.SetLineColor(kOrange+8);
    funZMuMuNoIso.SetLineWidth(3);
    //funZMuMuNoIso.SetLineStyle(kDashed);
    
    //funZMuMuNoIso.SetFillColor(kOrange-2);
    //funZMuMuNoIso.SetFillStyle(3325);

    funZMuMuNoIso.SetNpx(10000);
    TF1 funZMuMuNoIsoBkg = root::tf1_t<bkg_tag, Expr>("ZMuMuNoIsoBack", zMuMuNoIsoBkgScaled, fMin, fMax, 
						 yieldBkgZMuMuNotIso, alpha, b0, b1, b2);
    funZMuMuNoIsoBkg.SetLineColor(kViolet+3);
    funZMuMuNoIsoBkg.SetLineWidth(2);
    funZMuMuNoIsoBkg.SetLineStyle(kSolid);
    funZMuMuNoIsoBkg.SetFillColor(kViolet-5);
    funZMuMuNoIsoBkg.SetFillStyle(3357);

    funZMuMuNoIsoBkg.SetNpx(10000);
    histoZMuMuNoIso->SetTitle("Z -> #mu #mu Not Iso mass");
    histoZMuMuNoIso->SetXTitle("#mu +  #mu invariant mass (GeV/c^{2})");
    histoZMuMuNoIso->SetYTitle("Events");
    TCanvas *canvas = new TCanvas("canvas");
    histoZMuMuNoIso->Draw("e");
    funZMuMuNoIsoBkg.Draw("same");
    funZMuMuNoIso.Draw("same");
    canvas->SaveAs(ZMuMuNoIsoPlot.c_str());
    canvas->SetLogy();
    string logZMuMuNoIsoPlot = "log_" + ZMuMuNoIsoPlot;
    canvas->SaveAs(logZMuMuNoIsoPlot.c_str());
    
    double IntSigMMNotIso = ((double) rebinMuMu/  (double) rebinMuMuNoIso) * funZMuMuNoIso.Integral(fMin, fMax);
    double IntSigMMNotIsoBkg = ((double) rebinMuMu/  (double) rebinMuMuNoIso) * funZMuMuNoIsoBkg.Integral(fMin, fMax);
    cout << "*********  ZMuMuNoIsoPlot signal yield from the fit ==> " <<  IntSigMMNotIso << endl;
    cout << "*********  ZMuMuNoIsoPlot background yield from the fit ==> " << IntSigMMNotIsoBkg << endl;



    string ZMuTkPlot = "ZMuTkFit_X_" + plot_string;
    root::plot<Expr>(ZMuTkPlot.c_str(), *histoZMuTk, zMuTk, fMin, fMax,
		     effTk, effSa, effIso, effHLT, yieldZMuMu,
		     yieldBkgZMuTk, lambda, a0, a1, a2,
		     kOrange+3, 2, kSolid, 100,
		     "Z -> #mu + (unmatched) track mass", "#mu #mu invariant mass (GeV/c^{2})",
		     "Events");
    ZMuTkPlot = "ZMuTkFit_" + plot_string;
    TF1 funZMuTk = root::tf1_t<sig_tag, Expr>("ZMuTkFunction", zMuTk, fMin, fMax, 
					      effTk, effSa, effIso, effHLT, yieldZMuMu, 
					      yieldBkgZMuTk, lambda, a0, a1, a2);
    funZMuTk.SetLineColor(kOrange+8);
    funZMuTk.SetLineWidth(3);
    funZMuTk.SetLineStyle(kSolid);
    //  funZMuTk.SetFillColor(kOrange-2);
    //funZMuTk.SetFillStyle(3325);
    funZMuTk.SetNpx(10000);
    TF1 funZMuTkBkg = root::tf1_t<bkg_tag, Expr>("ZMuTkBack", zMuTkBkgScaled, fMin, fMax, 
						 yieldBkgZMuTk, lambda, a0, a1, a2);
    funZMuTkBkg.SetLineColor(kViolet+3);
    funZMuTkBkg.SetLineWidth(2);
    funZMuTkBkg.SetLineStyle(kSolid);
    funZMuTkBkg.SetFillColor(kViolet-5);
    funZMuTkBkg.SetFillStyle(3357);
    funZMuTkBkg.SetNpx(10000);
    histoZMuTk->SetTitle("Z -> #mu + (unmatched) track mass");
    histoZMuTk->SetXTitle("#mu + (unmatched) track invariant mass (GeV/c^{2})");
    histoZMuTk->SetYTitle("Events");
    TCanvas *canvas_ = new TCanvas("canvas_");
    histoZMuTk->Draw("e");
    funZMuTkBkg.Draw("same");
    funZMuTk.Draw("same");
    canvas_->SaveAs(ZMuTkPlot.c_str());
    canvas_->SetLogy();
    string logZMuTkPlot = "log_" + ZMuTkPlot;
    canvas_->SaveAs(logZMuTkPlot.c_str());


    double IntSigMT = ((double) rebinMuMu/  (double) rebinMuTk) * funZMuTk.Integral(fMin, fMax);
    double IntSigMTBkg = ((double) rebinMuMu/  (double) rebinMuTk) * funZMuTkBkg.Integral(fMin, fMax);
    cout << "*********  ZMuMuTkPlot signal yield from the fit ==> " <<  IntSigMT << endl;
    cout << "*********  ZMuMuTkPlot background yield from the fit ==> " << IntSigMTBkg << endl;



    string ZMuSaPlot = "ZMuSaFit_X_" + plot_string;
    root::plot<Expr>(ZMuSaPlot.c_str(), *histoZMuSa, zMuSa, fMin, fMax, 
		     effSa, effTk, effIso, yieldZMuMu, 
		     yieldBkgZMuSa, beta, c0, c1, c2 ,
		     kOrange+3, 2, kSolid, 100, 
		     "Z -> #mu + (unmatched) standalone mass", 
		     "#mu + (unmatched) standalone invariant mass (GeV/c^{2})", 
		     "Events");



    ZMuSaPlot = "ZMuSaFit_" + plot_string;
    TF1 funZMuSa = root::tf1_t<sig_tag, Expr>("ZMuSaFunction", zMuSa, fMin, fMax, 
					      effTk, effSa, effIso, effHLT, yieldZMuMu, 
					      yieldBkgZMuSa, beta, c0, c1, c2);
    funZMuSa.SetLineColor(kOrange+8);
    funZMuSa.SetLineWidth(3);
    funZMuSa.SetLineStyle(kSolid);
    // funZMuSa.SetFillColor(kOrange-2);
    // funZMuSa.SetFillStyle(3325);
    funZMuSa.SetNpx(10000);
    TF1 funZMuSaBkg = root::tf1_t<bkg_tag, Expr>("ZMuSaBack", zMuSaBkgScaled, fMin, fMax, 
						 yieldBkgZMuSa, beta, c0, c1, c2);
    funZMuSaBkg.SetLineColor(kViolet+3);
    funZMuSaBkg.SetLineWidth(2);
    funZMuSaBkg.SetLineStyle(kSolid);
    funZMuSaBkg.SetFillColor(kViolet-5);
    funZMuSaBkg.SetFillStyle(3357);
    funZMuSaBkg.SetNpx(10000);
    histoZMuSa->SetTitle("Z -> #mu + (unmatched) standalone mass");
    histoZMuSa->SetXTitle("#mu + (unmatched) standalone invariant mass (GeV/c^{2})");
    histoZMuSa->SetYTitle("Events");
    TCanvas *canvas__ = new TCanvas("canvas__");
    histoZMuSa->Draw("e");
    funZMuSaBkg.Draw("same");
    funZMuSa.Draw("same");
    canvas__->SaveAs(ZMuSaPlot.c_str());
    canvas__->SetLogy();
    string logZMuSaPlot = "log_" + ZMuSaPlot;
    canvas__->SaveAs(logZMuSaPlot.c_str());

    double IntSigMS = ((double) rebinMuMu/  (double) rebinMuSa) * funZMuSa.Integral(fMin, fMax);
    double IntSigMSBkg = ((double) rebinMuMu/  (double) rebinMuSa) * funZMuSaBkg.Integral(fMin, fMax);
    cout << "*********  ZMuMuSaPlot signal yield from the fit ==> " <<  IntSigMS << endl;
    cout << "*********  ZMuMuSaPlot background yield from the fit ==> " << IntSigMSBkg << endl;


  }
  return 0;
}

#include <boost/program_options.hpp>
using namespace boost;
namespace po = boost::program_options;

int main(int ac, char *av[]) {
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("input-file,i", po::value<vector<string> >(), "input file")
    ("min,m", po::value<double>(&fMin)->default_value(60), "minimum value for fit range")
    ("max,M", po::value<double>(&fMax)->default_value(120), "maximum value for fit range")
    ("rebins,R", po::value<vector<unsigned int> >(), "rebins values: rebinMuMu2HLT , rebinMuMu1HLT , rebinMuMuNoIso , rebinMuSa, rebinMuTk")
    ("chi2,c", "perform chi-squared fit")
    ("plr,p", "perform Poisson likelihood-ratio fit")
    ("nonIsoTemplateFromMC,I", po::value<bool>(&nonIsoTemplateFromMC)->default_value(false) , "take the template for nonIso sample from MC")
    ("plot-format,f", po::value<string>(&ext)->default_value("eps"), "output plot format")
    ("detectorRegion,r",po::value<string> (&region)->default_value("all"), "detector region in which muons are detected" );   
  po::positional_options_description p;
  p.add("input-file", -1);
  p.add("rebins", -1);

  
  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).
	    options(desc).positional(p).run(), vm);
  po::notify(vm);
  
  if (vm.count("help")) {
    cout << "Usage: options_description [options]\n";
    cout << desc;
    return 0;
  }
  
  if (!vm.count("input-file")) {
    return 1;
  }
  cout << "Input files are: " 
       << vm["input-file"].as< vector<string> >() << "\n";
  vector<string> v_file = vm["input-file"].as< vector<string> >();


  if (vm.count("rebins") ) {
    vector<unsigned int> v_rebin = vm["rebins"].as< vector<unsigned int> >();
    if (v_rebin.size()!=5){
      cerr << " please provide 5 numbers in the given order:  rebinMuMu2HLT , rebinMuMu1HLT , rebinMuMuNoIso, rebinMuSa, rebinMuTk \n";
      return 1;
    }
 rebinMuMuNoIso = v_rebin[2], rebinMuMu1HLT = v_rebin[1], rebinMuMu2HLT = v_rebin[0], rebinMuTk = v_rebin[4], rebinMuSa = v_rebin[3];
  }




  bool chi2Fit = vm.count("chi2"), plrFit = vm.count("plr");
  
  if(!(chi2Fit||plrFit))
    cerr << "Warning: no fit performed. Please, specify either -c or -p options or both" << endl;
 

  gROOT->SetStyle("Plain");

  int ret = 0;
  try {
    if(plrFit) {
      std::cout << "==================================== " << std::endl;
      std::cout << "=== Poisson Likelihood Ratio fit === " << std::endl;
      std::cout << "==================================== " << std::endl;
      int ret2 = main_t<ExprPLR>(v_file);
      if(ret2 != 0) ret = 1;
    }
    if(chi2Fit) {
      std::cout << "================= " << std::endl;
      std::cout << "=== Chi-2 fit === " << std::endl;
      std::cout << "================= " << std::endl;
      int ret1 = main_t<ExprChi2>(v_file);
      if(ret1 != 0) ret = 1;
    }
  }
  catch(std::exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    cerr << "Exception of unknown type!\n";
  }
  return ret;
}

