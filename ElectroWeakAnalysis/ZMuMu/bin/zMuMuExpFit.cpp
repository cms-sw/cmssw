#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/ZLineShape.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include "PhysicsTools/Utilities/interface/Exponential.h"
#include "PhysicsTools/Utilities/interface/Polynomial.h"
#include "PhysicsTools/Utilities/interface/Convolution.h"
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/MultiHistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/RootMinuitCommands.h"
#include "PhysicsTools/Utilities/interface/rootPlot.h"
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

typedef funct::GaussIntegrator IntegratorConv;

int main(int ac, char *av[]) {
  gROOT->SetStyle("Plain");
  try {
    typedef funct::Product<funct::Exponential, 
                           funct::Convolution<funct::ZLineShape, funct::Gaussian, IntegratorConv>::type >::type ZPeak;
    typedef funct::Product<funct::Parameter, ZPeak>::type ZMuMuSig;
    typedef ZMuMuSig ZMuMu;
    typedef ZMuMuSig ZMuTkSig;
    typedef funct::Product<funct::Parameter, 
                           funct::Product<funct::Exponential, funct::Polynomial<2> >::type >::type ZMuTkBkg;
    typedef funct::Sum<ZMuTkSig, ZMuTkBkg>::type ZMuTk;
    typedef funct::Product<funct::Parameter, funct::Gaussian>::type ZMuSaSig;
    typedef funct::Parameter ZMuSaBkg;
    typedef funct::Sum<ZMuSaSig, ZMuSaBkg>::type ZMuSa;
    
    typedef fit::MultiHistoChiSquare<ZMuMu, ZMuTk, ZMuSa> ChiSquared;
    fit::RootMinuitCommands<ChiSquared> commands("ElectroWeakAnalysis/ZMuMu/test/zMuMuExpFit.txt");
    
    double fMin, fMax;
    string ext;
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help message")
      ("include-path,I", po::value< vector<string> >(), 
       "include path")
      ("input-file", po::value< vector<string> >(), "input file")
      ("min,m", po::value<double>(&fMin)->default_value(60), "minimum value for fit range")
      ("max,M", po::value<double>(&fMax)->default_value(120), "maximum value for fit range")
      ("output-file,O", po::value<string>(&ext)->default_value(".ps"), 
       "output file format")
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
    
    if (vm.count("include-path")) {
      cout << "Include paths are: " 
	   << vm["include-path"].as< vector<string> >() << "\n";
    }
    
    if (vm.count("input-file")) {
      cout << "Input files are: " 
	   << vm["input-file"].as< vector<string> >() << "\n";
      vector<string> v_file = vm["input-file"].as< vector<string> >();
      for(vector<string>::const_iterator it = v_file.begin(); 
	  it != v_file.end(); ++it) {
	TFile * root_file = new TFile(it->c_str(),"read");
	TH1D * histoZMuMu = (TH1D*) root_file->Get("zToMM");
	fix(histoZMuMu);
	TH1D * histoZMuTk = (TH1D*) root_file->Get("zToMTk");
	fix(histoZMuTk);
	TH1D * histoZMuSa = (TH1D*) root_file->Get("zToMS");
	fix(histoZMuSa);
	cout << ">>> histogram loaded\n";
	string f_string = *it;
	replace(f_string.begin(), f_string.end(), '.', '_');
	string plot_string = f_string + ext;
	cout << ">>> Input files loaded\n";
	
	const char * kYieldZMuMu = "YieldZMuMu";
	const char * kYieldZMuTk = "YieldZMuTk";
	const char * kYieldZMuSa = "YieldZMuSa";
	const char * kYieldBkgZMuTk = "YieldBkgZMuTk"; 
	const char * kYieldBkgZMuSa = "YieldBkgZMuSa"; 
	const char * kLambdaZMuMu = "LambdaZMuMu";
	const char * kMass = "Mass";
	const char * kGamma = "Gamma";
	const char * kPhotonFactorZMuMu = "PhotonFactorZMuMu";
	const char * kInterferenceFactorZMuMu = "InterferenceFactorZMuMu";
	//const char * kPhotonFactorZMuTk = "PhotonFactorZMuTk";
	//const char * kInterferenceFactorZMuTk = "InterferenceFactorZMuTk";
	const char * kMeanZMuMu = "MeanZMuMu";
	const char * kSigmaZMuMu = "SigmaZMuMu";
	const char * kLambda = "Lambda";
	const char * kA0 = "A0"; 
	const char * kA1 = "A1"; 
	const char * kA2 = "A2"; 
	const char * kSigmaZMuSa = "SigmaZMuSa";

	IntegratorConv integratorConv(1.e-5);
	
	funct::Parameter lambdaZMuMu(kLambdaZMuMu, commands.par(kLambdaZMuMu));
	funct::Parameter mass(kMass, commands.par(kMass));
	funct::Parameter gamma(kGamma, commands.par(kGamma));
	funct::Parameter photonFactorZMuMu(kPhotonFactorZMuMu, commands.par(kPhotonFactorZMuMu)); 
	funct::Parameter interferenceFactorZMuMu(kInterferenceFactorZMuMu, commands.par(kInterferenceFactorZMuMu)); 
	//funct::Parameter photonFactorZMuTk(kPhotonFactorZMuTk, commands.par(kPhotonFactorZMuTk)); 
	//funct::Parameter interferenceFactorZMuTk(kInterferenceFactorZMuTk, commands.par(kInterferenceFactorZMuTk)); 
	funct::Parameter yieldZMuMu(kYieldZMuMu, commands.par(kYieldZMuMu));
	funct::Parameter yieldZMuTk(kYieldZMuTk, commands.par(kYieldZMuTk)); 
	funct::Parameter yieldZMuSa(kYieldZMuSa, commands.par(kYieldZMuSa)); 
	funct::Parameter yieldBkgZMuTk(kYieldBkgZMuTk, commands.par(kYieldBkgZMuTk));
	funct::Parameter yieldBkgZMuSa(kYieldBkgZMuSa, commands.par(kYieldBkgZMuSa));
	funct::Parameter meanZMuMu(kMeanZMuMu, commands.par(kMeanZMuMu));
	funct::Parameter sigmaZMuMu(kSigmaZMuMu, commands.par(kSigmaZMuMu)); 
	funct::Parameter sigmaZMuSa(kSigmaZMuSa, commands.par(kSigmaZMuSa)); 
	funct::Parameter lambda(kLambda, commands.par(kLambda));
	funct::Parameter a0(kA0, commands.par(kA0));
	funct::Parameter a1(kA1, commands.par(kA1));
	funct::Parameter a2(kA2, commands.par(kA2));
	                
	ZPeak zPeak = funct::Exponential(lambdaZMuMu) * 
	              funct::conv(funct::ZLineShape(mass, gamma, photonFactorZMuMu, interferenceFactorZMuMu), 
			 	  funct::Gaussian(meanZMuMu, sigmaZMuMu), 
				  -3*sigmaZMuMu.value(), 3*sigmaZMuMu.value(), integratorConv);
	ZMuMu zMuMu = yieldZMuMu * zPeak;
	ZMuTkBkg zMuTkBkg = 
	  yieldBkgZMuTk * (funct::Exponential(lambda) * funct::Polynomial<2>(a0, a1, a2));
	ZMuTk zMuTk = yieldZMuTk * zPeak + zMuTkBkg;
	ZMuSa zMuSa = yieldZMuSa * funct::Gaussian(mass, sigmaZMuSa) + yieldBkgZMuSa;
	
	ChiSquared chi2(zMuMu, histoZMuMu, 
			zMuTk, histoZMuTk, 
			zMuSa, histoZMuSa, 
			fMin, fMax);
	cout << "N. deg. of freedom: " << chi2.numberOfBins() << endl;
	fit::RootMinuit<ChiSquared> minuit(chi2, true);
	commands.add(minuit, yieldZMuMu);
	commands.add(minuit, yieldZMuTk);
	commands.add(minuit, yieldZMuSa);
	commands.add(minuit, yieldBkgZMuTk);
	commands.add(minuit, yieldBkgZMuSa);
	commands.add(minuit, lambdaZMuMu);
	commands.add(minuit, mass);
	commands.add(minuit, gamma);
	commands.add(minuit, photonFactorZMuMu);
	commands.add(minuit, interferenceFactorZMuMu);
	//commands.add(minuit, photonFactorZMuTk);
	//commands.add(minuit, interferenceFactorZMuTk);
	commands.add(minuit, meanZMuMu);
	commands.add(minuit, sigmaZMuMu);
	commands.add(minuit, lambda);
	commands.add(minuit, a0);
	commands.add(minuit, a1);
	commands.add(minuit, a2);
	commands.add(minuit, sigmaZMuSa);
	commands.run(minuit);
	const unsigned int nPar = 17;
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
	string ZMuMuPlot = "ZMuMuFit" + plot_string;
	root::plot<ZMuMu>(ZMuMuPlot.c_str(), *histoZMuMu, zMuMu, fMin, fMax, 
			  yieldZMuMu, lambdaZMuMu, mass, gamma, photonFactorZMuMu, interferenceFactorZMuMu, 
			  meanZMuMu, sigmaZMuMu, 
			  kRed, 2, kDashed, 10000, 
			  "Z -> #mu #mu mass with isolation cut", "#mu #mu invariant mass (GeV/c^{2})", 
			  "Events");
	string ZMuTkPlot = "ZMuTkFit" + plot_string;
	TF1 funZMuTk = root::tf1<ZMuTk>("ZMuTkFunction", zMuTk, fMin, fMax, 
					yieldZMuTk, lambdaZMuMu, mass, gamma, photonFactorZMuMu, interferenceFactorZMuMu, 
					meanZMuMu, sigmaZMuMu, 
					yieldBkgZMuTk, lambda, a0, a1, a2);
	funZMuTk.SetLineColor(kRed);
	funZMuTk.SetLineWidth(2);
	funZMuTk.SetLineStyle(kDashed);
	funZMuTk.SetNpx(10000);
	TF1 funZMuTkBkg = root::tf1<ZMuTkBkg>("ZMuTkBack", zMuTkBkg, fMin, fMax, 
					      yieldBkgZMuTk, lambda, a0, a1, a2);
	funZMuTkBkg.SetLineColor(kGreen);
	funZMuTkBkg.SetLineWidth(2);
	funZMuTkBkg.SetLineStyle(kDashed);
	funZMuTkBkg.SetNpx(10000);
	histoZMuTk->SetTitle("Z -> #mu + (unmatched) track mass with isolation cut");
	histoZMuTk->SetXTitle("#mu + (unmatched) track invariant mass (GeV/c^{2})");
	histoZMuTk->SetYTitle("Events");
	TCanvas *canvas = new TCanvas("canvas");
	histoZMuTk->Draw("e");
	funZMuTk.Draw("same");
	funZMuTkBkg.Draw("same");
	canvas->SaveAs(ZMuTkPlot.c_str());
	canvas->SetLogy();
	string logZMuTkPlot = "log_" + ZMuTkPlot;
	canvas->SaveAs(logZMuTkPlot.c_str());
	string ZMuSaPlot = "ZMuSaFit" + plot_string;
	root::plot<ZMuSa>(ZMuSaPlot.c_str(), *histoZMuSa, zMuSa, fMin, fMax, 
			  yieldZMuSa, mass, sigmaZMuSa, yieldBkgZMuSa, 
			  kRed, 2, kDashed, 10000, 
			  "Z -> #mu + (unmatched) standalone mass with isolation cut", 
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

