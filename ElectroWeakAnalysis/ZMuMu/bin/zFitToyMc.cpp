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
#include <boost/program_options.hpp>
using namespace boost;
namespace po = boost::program_options;

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

int main(int ac, char *av[]) {
  gROOT->SetStyle("Plain");

  try {
    typedef funct::FunctExpression Expr;
    typedef fit::HistoChiSquare<funct::FunctExpression> ExprChi2;
    typedef fit::MultiHistoChiSquare<ExprChi2, ExprChi2, ExprChi2, ExprChi2, ExprChi2> ChiSquared;
    double fMin, fMax;
    string ext;
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help,h", "produce help message")
      ("input-file,i", po::value< vector<string> >(), "input file")
      ("min,m", po::value<double>(&fMin)->default_value(60), "minimum value for fit range")
      ("max,M", po::value<double>(&fMax)->default_value(120), "maximum value for fit range")
      ("plot-format,p", po::value<string>(&ext)->default_value("eps"), 
       "output plot format");
    
    po::positional_options_description p;
    p.add("input-file", -1);
    
    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).
	    options(desc).positional(p).run(), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
      cout << "Usage: options_description [options]\n";
      cout << desc;
      return 0;
    }
    
    fit::RootMinuitCommands<ChiSquared> commands("zFitToyMc.txt");

    cout << "minuit command file completed" << endl;
    const unsigned int rebinMuMuNoIso = 2,rebinMuMu = 1, rebinMuMu1HLT = 1, rebinMuMu2HLT = 1, rebinMuTk = 2, rebinMuSa = 10;
    // assume that the bin size is 1 GeV!!!
    funct::Constant rebinMuMuNoIsoConst(rebinMuMuNoIso), rebinMuMuConst(rebinMuMu), 
      rebinMuMu1HLTConst(rebinMuMu1HLT), rebinMuMu2HLTConst(rebinMuMu2HLT), 
      rebinMuTkConst(rebinMuTk), rebinMuSaConst(rebinMuSa);

    if (vm.count("input-file")) {
      cout << "Input files are: " 
	   << vm["input-file"].as< vector<string> >() << "\n";
      vector<string> v_file = vm["input-file"].as< vector<string> >();
      for(vector<string>::const_iterator it = v_file.begin(); 
	  it != v_file.end(); ++it) {
	TFile * root_file = new TFile(it->c_str(), "read");
	TH1 * histoZMuMuNoIso = getHisto(root_file, "nonIsolatedZToMuMuPlots/zMass_noIso",rebinMuMuNoIso);
	TH1 * histoZMuMu = getHisto(root_file, "goodZToMuMuPlots/zMass_golden",rebinMuMu);
	TH1 * histoZMuMu1HLT = getHisto(root_file, "goodZToMuMu1HLTPlots/zMass_1hlt", rebinMuMu1HLT);
	TH1 * histoZMuMu2HLT = getHisto(root_file, "goodZToMuMu2HLTPlots/zMass_2hlt", rebinMuMu2HLT);
	TH1 * histoZMuTk = getHisto(root_file, "goodZToMuMuOneTrackPlots/zMass_tk", rebinMuTk);
	TH1 * histoZMuSa = getHisto(root_file, "goodZToMuMuOneStandAloneMuonPlots/zMass_sa", rebinMuSa);
	TH1 * histoZMuSaFromMuMu = getHisto(root_file, "zmumuSaMassHistogram/zMass_safromGolden", rebinMuSa);


	cout << ">>> histogram loaded\n";
	string f_string = *it;
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
	funct::RootHistoPdf zPdfMuMuNonIso(*histoZMuMu, fMin, fMax);//imposto le pdf a quella di ZMuMu
	funct::RootHistoPdf zPdfMuTk = zPdfMuMuNonIso;
	funct::RootHistoPdf zPdfMuMu1HLT = zPdfMuMuNonIso;
	funct::RootHistoPdf zPdfMuMu2HLT = zPdfMuMuNonIso;
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
	Expr zMuMuNoIsoEffTerm = (effTk ^ _2) * (effSa ^ _2) * (_1 - (effIso ^ _2)) * (_1 - ((_1 - effHLT)^_2));
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

	Expr zMuSa = rebinMuSaConst * (zMuSaEffTerm * yieldZMuMu * zPdfMuSa); // + (yieldBkgZMuSa * funct::Exponential(beta) )); 

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
	commands.run(minuit);
	const unsigned int nPar = 17;//WARNIG: this must be updated manually for now
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
			  kRed, 2, kDashed, 100, 
			  "Z -> #mu #mu mass", "#mu #mu invariant mass (GeV/c^{2})", 
			  "Events");

	string ZMuMu2HLTPlot = "ZMuMu2HLTFit_" + plot_string;
	root::plot<Expr>(ZMuMu2HLTPlot.c_str(), *histoZMuMu2HLT, zMuMu2HLT, fMin, fMax, 
			  effTk, effSa, effIso, effHLT, yieldZMuMu, 
			  kRed, 2, kDashed, 100, 
			  "Z -> #mu #mu mass", "#mu #mu invariant mass (GeV/c^{2})", 
			  "Events");

	
	string ZMuMuNoIsoPlot = "ZMuMuNoIsoFit_" + plot_string;
	root::plot<Expr>(ZMuMuNoIsoPlot.c_str(), *histoZMuMuNoIso, zMuMuNoIso, fMin, fMax, 
			 effTk, effSa, effIso, effHLT, yieldZMuMu,
			 kRed, 2, kDashed, 100, 
			 "Z -> #mu #mu Not Iso mass", "#mu #mu invariant mass (GeV/c^{2})", 
			 "Events");	
	
	string ZMuTkPlot = "ZMuTkFit_X_" + plot_string;
	root::plot<Expr>(ZMuTkPlot.c_str(), *histoZMuTk, zMuTk, fMin, fMax,
			 effTk, effSa, effIso, effHLT, yieldZMuMu,
			 yieldBkgZMuTk, lambda, a0, a1, a2,
			 kRed, 2, kDashed, 100,
                         "Z -> #mu + (unmatched) track mass", "#mu #mu invariant mass (GeV/c^{2})",
                         "Events");
	ZMuTkPlot = "ZMuTkFit_" + plot_string;
	TF1 funZMuTk = root::tf1_t<sig_tag, Expr>("ZMuTkFunction", zMuTk, fMin, fMax, 
						  effTk, effSa, effIso, effHLT, yieldZMuMu, 
						  yieldBkgZMuTk, lambda, a0, a1, a2);
	funZMuTk.SetLineColor(kRed);
	funZMuTk.SetLineWidth(2);
	funZMuTk.SetLineStyle(kDashed);
	funZMuTk.SetNpx(10000);
	TF1 funZMuTkBkg = root::tf1_t<bkg_tag, Expr>("ZMuTkBack", zMuTkBkgScaled, fMin, fMax, 
						     yieldBkgZMuTk, lambda, a0, a1, a2);
	funZMuTkBkg.SetLineColor(kGreen);
	funZMuTkBkg.SetLineWidth(2);
	funZMuTkBkg.SetLineStyle(kDashed);
	funZMuTkBkg.SetNpx(10000);
	histoZMuTk->SetTitle("Z -> #mu + (unmatched) track mass");
	histoZMuTk->SetXTitle("#mu + (unmatched) track invariant mass (GeV/c^{2})");
	histoZMuTk->SetYTitle("Events");
	TCanvas *canvas = new TCanvas("canvas");
	histoZMuTk->Draw("e");
	funZMuTkBkg.Draw("same");
	funZMuTk.Draw("same");
	canvas->SaveAs(ZMuTkPlot.c_str());
	canvas->SetLogy();
	string logZMuTkPlot = "log_" + ZMuTkPlot;
	canvas->SaveAs(logZMuTkPlot.c_str());
	string ZMuSaPlot = "ZMuSaFit_" + plot_string;
	root::plot<Expr>(ZMuSaPlot.c_str(), *histoZMuSa, zMuSa, fMin, fMax, 
			 effSa, effTk, effIso,
			 yieldZMuMu, yieldBkgZMuSa, 
			 kRed, 2, kDashed, 10000, 
			 "Z -> #mu + (unmatched) standalone mass", 
			 "#mu + (unmatched) standalone invariant mass (GeV/c^{2})", 
			 "Events");
      }
    }
  }
  catch(std::exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    cerr << "Exception of unknown type!\n";
  }

  return 0;
}



