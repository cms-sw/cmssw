#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include "PhysicsTools/Utilities/interface/Numerical.h"
#include "PhysicsTools/Utilities/interface/Exponential.h"
#include "PhysicsTools/Utilities/interface/Polynomial.h"
#include "PhysicsTools/Utilities/interface/Constant.h"
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/MultiHistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/RootMinuitCommands.h"
#include "PhysicsTools/Utilities/interface/FunctClone.h"
#include "PhysicsTools/Utilities/interface/rootPlot.h"
#include "PhysicsTools/Utilities/interface/Expression.h"
#include "PhysicsTools/Utilities/interface/HistoPdf.h"
#include "TROOT.h"
#include "TH1.h"
#include "TFile.h"
#include <boost/program_options.hpp>
using namespace boost;
namespace po = boost::program_options;

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

//A function that sets istogram contents to 0 
//if they are too small
void fix(TH1* histo) {
  for(int i = 1; i <= histo->GetNbinsX(); ++i) {
    if(histo->GetBinContent(i) < 0.1) {
      histo->SetBinContent(i, 0.0);
      histo->SetBinError(i, 0.0);
    }
  }
}

typedef funct::FunctExpression Expr;
typedef fit::MultiHistoChiSquare<Expr, Expr, Expr, Expr> ChiSquared;

struct sig_tag;
struct bkg_tag;

int main(int ac, char *av[]) {
  gROOT->SetStyle("Plain");
  try {

    double fMin, fMax;
    string ext, variable, muCharge;
    int binNumber;
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help,h", "produce help message")
      ("input-file,i", po::value< vector<string> >(), "input file")
      ("min,m", po::value<double>(&fMin)->default_value(60), "minimum value for fit range")
      ("max,M", po::value<double>(&fMax)->default_value(120), "maximum value for fit range")
      ("eta_or_pt,v", po::value<string>(&variable)->default_value("eta"), "variable to study (eta or pt)") 
      ("charge,q", po::value<string>(&muCharge)->default_value("minus"),"muon charge to study (minus or plus)")
      ("binNum,b", po::value<int>(&binNumber)->default_value(0), "cynematic bin to fit")
      ("plot-format,p", po::value<string>(&ext)->default_value("ps"), 
       "output plot format")
      ;
    
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
    
    fit::RootMinuitCommands<ChiSquared> commands("csa08NewZFit.txt");

    const unsigned int rebinMuMu = 1, rebinMuTk = 2, rebinMuSa = 1;
    // assume that the bin size is 1 GeV!!!
    funct::Constant rebinMuMuConst(rebinMuMu), rebinMuTkConst(rebinMuTk), rebinMuSaConst(rebinMuSa);

    if (vm.count("input-file")) {
      cout << "Input files are: " 
	   << vm["input-file"].as< vector<string> >() << "\n";
      vector<string> v_file = vm["input-file"].as< vector<string> >();
      for(vector<string>::const_iterator it = v_file.begin(); 
	  it != v_file.end(); ++it) {
	TFile * root_file = new TFile(it->c_str(),"read");
	cout <<"start " << endl;
	// variable and charge definition at moment in manual way
	//	string variable = "eta";
	//	string muCharge = "minus";
	////////////////////////////////////////

	stringstream sslabelZMuMu2HLT; 
	sslabelZMuMu2HLT << "zMuMu_efficiencyAnalyzer/" << variable << "Intervals" << "/zmumu2HLT" << muCharge << "_" << variable << "Range" << binNumber;
	stringstream sslabelZMuMu1HLT; 
	sslabelZMuMu1HLT << "zMuMu_efficiencyAnalyzer/" << variable << "Intervals" << "/zmumu1HLT" << muCharge << "_" << variable << "Range" << binNumber;
	stringstream sslabelZMuTk; 
	sslabelZMuTk << "zMuMu_efficiencyAnalyzer/" << variable << "Intervals" << "/zmutrack" << muCharge << "_" << variable << "Range" << binNumber;
	stringstream sslabelZMuSa; 
	sslabelZMuSa << "zMuMu_efficiencyAnalyzer/" << variable << "Intervals" << "/zmusta" << muCharge << "_" << variable << "Range" << binNumber;

	cout << "histoZMuMu2HLT:   " << sslabelZMuMu2HLT.str() << endl;
	TH1D * histoZMuMu2HLT = (TH1D*) root_file->Get(sslabelZMuMu2HLT.str().c_str());
	histoZMuMu2HLT->Rebin(rebinMuMu);
	fix(histoZMuMu2HLT);
	cout << "histoZMuMu1HLT:   " << sslabelZMuMu1HLT.str() << endl;
	TH1D * histoZMuMu1HLT = (TH1D*) root_file->Get(sslabelZMuMu1HLT.str().c_str());
	histoZMuMu1HLT->Rebin(rebinMuMu);
	fix(histoZMuMu1HLT);
	cout << "histoZMuTk:   " << sslabelZMuTk.str() << endl;
	TH1D * histoZMuTk = (TH1D*) root_file->Get(sslabelZMuTk.str().c_str());
	//	histoZMuTk->Rebin(rebinMuTk);
	fix(histoZMuTk);
	cout << "histoZMuSa:   " << sslabelZMuSa.str() << endl;
	TH1D * histoZMuSa = (TH1D*) root_file->Get(sslabelZMuSa.str().c_str());
	//	histoZMuSa->Rebin(rebinMuSa);
	fix(histoZMuSa);
	cout << ">>> histogram loaded\n";

	string f_string = *it;
	replace(f_string.begin(), f_string.end(), '.', '_');
	replace(f_string.begin(), f_string.end(), '/', '_');
	string plot_string = f_string + "." + ext;
	cout << ">>> Input files loaded\n" << f_string << endl;
		
	const char * kYieldZMuMu = "YieldZMuMu";
	const char * kEfficiencyHLT = "EfficiencyHLT";
	const char * kEfficiencyTk = "EfficiencyTk";
	const char * kEfficiencySa = "EfficiencySa";
	const char * kYieldBkgZMuTk = "YieldBkgZMuTk"; 
	const char * kBeta = "Beta";
	const char * kLambda = "Lambda";
	//	const char * kA0 = "A0"; 
	//	const char * kA1 = "A1"; 
	//	const char * kA2 = "A2"; 

	funct::Parameter yieldZMuMu(kYieldZMuMu, commands.par(kYieldZMuMu));
	funct::Parameter effHLT(kEfficiencyHLT, commands.par(kEfficiencyHLT)); 
	funct::Parameter effTk(kEfficiencyTk, commands.par(kEfficiencyTk)); 
	funct::Parameter effSa(kEfficiencySa, commands.par(kEfficiencySa)); 
	funct::Parameter yieldBkgZMuTk(kYieldBkgZMuTk, commands.par(kYieldBkgZMuTk));
	funct::Parameter beta(kBeta, commands.par(kBeta));
	funct::Parameter lambda(kLambda, commands.par(kLambda));
	//	funct::Parameter a0(kA0, commands.par(kA0));
	//	funct::Parameter a1(kA1, commands.par(kA1));
	//	funct::Parameter a2(kA2, commands.par(kA2));
	funct::Constant cFMin(fMin), cFMax(fMax);

	// add zMuMu2HLT and zMuMu1HLT to build pdf
	TH1D *histoZMuMu = (TH1D *) histoZMuMu2HLT->Clone();
	histoZMuMu->Sumw2(); 
	histoZMuMu->Add(histoZMuMu2HLT,histoZMuMu1HLT); 

	// count ZMuMu Yield
	double nZMuMu = 0;
	{
	  unsigned int nBins = histoZMuMu->GetNbinsX();
	  double xMin = histoZMuMu->GetXaxis()->GetXmin();
	  double xMax = histoZMuMu->GetXaxis()->GetXmax();
	  double deltaX =(xMax - xMin) / nBins;
	  for(unsigned int i = 0; i < nBins; ++i) { 
	    double x = xMin + (i +.5) * deltaX;
	    if(x > fMin && x < fMax)
	      nZMuMu += histoZMuMu->GetBinContent(i+1);
	  }
	}

	// count ZMuMu2HLT Yield
	double nZMuMu2HLT = 0;
	{
	  unsigned int nBins = histoZMuMu2HLT->GetNbinsX();
	  double xMin = histoZMuMu2HLT->GetXaxis()->GetXmin();
	  double xMax = histoZMuMu2HLT->GetXaxis()->GetXmax();
	  double deltaX =(xMax - xMin) / nBins;
	  for(unsigned int i = 0; i < nBins; ++i) { 
	    double x = xMin + (i +.5) * deltaX;
	    if(x > fMin && x < fMax)
	      nZMuMu2HLT += histoZMuMu2HLT->GetBinContent(i+1);
	  }
	}

	// count ZMuMu1HLT Yield
	double nZMuMu1HLT = 0;
	{
	  unsigned int nBins = histoZMuMu1HLT->GetNbinsX();
	  double xMin = histoZMuMu1HLT->GetXaxis()->GetXmin();
	  double xMax = histoZMuMu1HLT->GetXaxis()->GetXmax();
	  double deltaX =(xMax - xMin) / nBins;
	  for(unsigned int i = 0; i < nBins; ++i) { 
	    double x = xMin + (i +.5) * deltaX;
	    if(x > fMin && x < fMax)
	      nZMuMu1HLT += histoZMuMu1HLT->GetBinContent(i+1);
	  }
	}

	// count ZMuSa Yield (too low statistis so we just check the number assuming 0 background)
	double nZMuSa = 0;
	{
	  unsigned int nBins = histoZMuSa->GetNbinsX();
	  double xMin = histoZMuSa->GetXaxis()->GetXmin();
	  double xMax = histoZMuSa->GetXaxis()->GetXmax();
	  double deltaX =(xMax - xMin) / nBins;
	  for(unsigned int i = 0; i < nBins; ++i) { 
	    double x = xMin + (i +.5) * deltaX;
	    if(x > fMin && x < fMax)
	      nZMuSa += histoZMuSa->GetBinContent(i+1);
	  }
	}

	cout << ">>> count of ZMuMu yield in the range [" << fMin << ", " << fMax << "]: " << nZMuMu << endl;
	cout << ">>> count of ZMuMu2HLT yield in the range [" << fMin << ", " << fMax << "]: " << nZMuMu2HLT << endl;
	cout << ">>> count of ZMuMu1HLT yield in the range [" << fMin << ", " << fMax << "]: " << nZMuMu1HLT << endl;
	cout << ">>> count of ZMuSa yield in the range [" << fMin << ", " << fMax << "]: " << nZMuSa << endl;
	

	funct::RootHistoPdf zPdfMuTk(*histoZMuMu, fMin, fMax);
	zPdfMuTk.rebin(rebinMuTk);

	funct::Numerical<2> _2;
	funct::Numerical<1> _1;

	Expr zMuMu2HLTEffTerm = effTk *  effSa * effHLT; 
	Expr zMuMu1HLTEffTerm = effTk *  effSa * (_1 - effHLT); 
	Expr zMuTkEffTerm = effTk * (_1 - effSa);
	Expr zMuSaEffTerm = effSa * (_1 - effTk);

	Expr zMuMu2HLT = rebinMuMuConst * zMuMu2HLTEffTerm * yieldZMuMu;
	Expr zMuMu1HLT = rebinMuMuConst * zMuMu1HLTEffTerm * yieldZMuMu;

	Expr zMuTkBkg = yieldBkgZMuTk * funct::Exponential(lambda); 
	  //* funct::Polynomial<2>(a0, a1, a2);
	Expr zMuTkBkgScaled = rebinMuTkConst * zMuTkBkg;
	Expr zMuTk = rebinMuTkConst * (zMuTkEffTerm * yieldZMuMu * zPdfMuTk + zMuTkBkg);
	Expr zMuSa = rebinMuSaConst * zMuSaEffTerm * yieldZMuMu;

	TH1D histoZMM2HLTCount("histoZMM2HLTCount", "", 1, fMin, fMax);
	histoZMM2HLTCount.Fill(100, nZMuMu2HLT);
	TH1D histoZMM1HLTCount("histoZMM1HLTCount", "", 1, fMin, fMax);
	histoZMM1HLTCount.Fill(100, nZMuMu1HLT);
	TH1D histoZMSCount("histoZMSCount", "", 1, fMin, fMax);
	histoZMSCount.Fill(100, nZMuSa);
				       
	ChiSquared chi2(zMuMu2HLT, & histoZMM2HLTCount, 
			zMuMu1HLT, & histoZMM1HLTCount,
			zMuTk, histoZMuTk, 
			zMuSa,  & histoZMSCount, 
			fMin, fMax);
	cout << "N. deg. of freedom: " << chi2.degreesOfFreedom() << endl;
	fit::RootMinuit<ChiSquared> minuit(chi2, true);
	commands.add(minuit, yieldZMuMu);
	commands.add(minuit, effHLT);
	commands.add(minuit, effTk);
	commands.add(minuit, effSa);
	commands.add(minuit, yieldBkgZMuTk);
	commands.add(minuit, lambda);
	commands.add(minuit, beta);
	//	commands.add(minuit, a0);
	//	commands.add(minuit, a1);
	//	commands.add(minuit, a2);
	commands.run(minuit);
	const unsigned int nPar = 7;//WARNING: this must be updated manually for now
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

	
	double s;
	s = 0;
	for(int i = 1; i <= histoZMuMu2HLT->GetNbinsX(); ++i)
	  s += histoZMuMu2HLT->GetBinContent(i);
	histoZMuMu2HLT->SetEntries(s);
	s = 0;
	for(int i = 1; i <= histoZMuMu1HLT->GetNbinsX(); ++i)
	  s += histoZMuMu1HLT->GetBinContent(i);
	histoZMuMu1HLT->SetEntries(s);
	s = 0;
	for(int i = 1; i <= histoZMuTk->GetNbinsX(); ++i)
	  s += histoZMuTk->GetBinContent(i);
	histoZMuTk->SetEntries(s);
	s = 0;
	for(int i = 1; i <= histoZMuSa->GetNbinsX(); ++i)
	  s += histoZMuSa->GetBinContent(i);
	histoZMuSa->SetEntries(s);
	stringstream mybin;
	mybin << muCharge << "_" << variable << binNumber << "_";
	string ZMuMu2HLTPlot = "ZMuMu2HLTFit_muon" + mybin.str() + plot_string;
	root::plot<Expr>(ZMuMu2HLTPlot.c_str(), *histoZMuMu2HLT, zMuMu2HLT, fMin, fMax, 
			  effHLT, effTk, effSa, yieldZMuMu, 
			  kRed, 2, kDashed, 100, 
			  "Z -> #mu #mu mass (2HLT)", "#mu #mu invariant mass (GeV/c^{2})", 
			  "Events");

	string ZMuMu1HLTPlot = "ZMuMu1HLTFit_muon" + mybin.str() + plot_string;
	root::plot<Expr>(ZMuMu1HLTPlot.c_str(), *histoZMuMu1HLT, zMuMu1HLT, fMin, fMax, 
			  effHLT, effTk, effSa, yieldZMuMu, 
			  kRed, 2, kDashed, 100, 
			  "Z -> #mu #mu mass (1HLT)", "#mu #mu invariant mass (GeV/c^{2})", 
			  "Events");
		
	string ZMuTkPlot = "ZMuTkFit_muon" + mybin.str() + plot_string;
	root::plot<Expr>(ZMuTkPlot.c_str(), *histoZMuTk, zMuTk, fMin, fMax,
			 effHLT, effTk, effSa, yieldZMuMu,
			 yieldBkgZMuTk, lambda, 
			 //a0, a1, a2,
			 kRed, 2, kDashed, 100,
                         "Z -> #mu + (unmatched) track mass", "#mu #mu invariant mass (GeV/c^{2})",
                         "Events");
	//	string ZMuTkPlot = "ZMuTkFit_muon" + muCharge + variable + binNumber + plot_string;
	TF1 funZMuTk = root::tf1_t<sig_tag, Expr>("ZMuTkFunction", zMuTk, fMin, fMax, 
						  effHLT, effTk, effSa, yieldZMuMu, 
						  yieldBkgZMuTk, lambda);
	funZMuTk.SetLineColor(kRed);
	funZMuTk.SetLineWidth(2);
	funZMuTk.SetLineStyle(kDashed);
	funZMuTk.SetNpx(10000);
	TF1 funZMuTkBkg = root::tf1_t<bkg_tag, Expr>("ZMuTkBack", zMuTkBkgScaled, fMin, fMax, 
						     yieldBkgZMuTk, lambda);
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
	string ZMuSaPlot = "ZMuSaFit_muon" + mybin.str() + plot_string;
	root::plot<Expr>(ZMuSaPlot.c_str(), *histoZMuSa, zMuSa, fMin, fMax, 
			 effHLT, effSa, effTk, yieldZMuMu, 
			 kRed, 2, kDashed, 10000, 
			 "Z -> #mu + (unmatched) standalone mass", 
			 "#mu + (unmatched) standalone invariant mass (GeV/c^{2})", 
			 "Events");
	
      }
	
    }
  }
  catch(exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    cerr << "Exception of unknown type!\n";
  }

  return 0;
}



