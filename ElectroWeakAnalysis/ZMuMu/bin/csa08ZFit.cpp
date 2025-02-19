#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/ZLineShape.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include "PhysicsTools/Utilities/interface/Numerical.h"
#include "PhysicsTools/Utilities/interface/Exponential.h"
#include "PhysicsTools/Utilities/interface/Polynomial.h"
#include "PhysicsTools/Utilities/interface/Constant.h"
#include "PhysicsTools/Utilities/interface/Convolution.h"
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/Integral.h"
#include "PhysicsTools/Utilities/interface/MultiHistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/RootMinuitCommands.h"
#include "PhysicsTools/Utilities/interface/FunctClone.h"
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
typedef funct::GaussIntegrator IntegratorNorm;
//typedef funct::TrapezoidIntegrator IntegratorConv;
//typedef funct::TrapezoidIntegrator IntegratorNorm;

typedef funct::Product<funct::Exponential, 
		       funct::Convolution<funct::ZLineShape, funct::Gaussian, IntegratorConv>::type>::type ZPeakNoNorm;

NUMERICAL_FUNCT_INTEGRAL(ZPeakNoNorm, IntegratorNorm);

typedef funct::DefIntegral<ZPeakNoNorm, funct::Constant, funct::Constant, IntegratorNorm> ZPeakNormFactor;
typedef funct::Ratio<ZPeakNoNorm, ZPeakNormFactor>::type ZPeak;

int main(int ac, char *av[]) {
  gROOT->SetStyle("Plain");
  try {
    typedef funct::Power<funct::Parameter, funct::Numerical<2> >::type IsoefficiencytermSQ;
    typedef funct::Master<funct::Product<funct::Parameter, ZPeak>::type> ZMuMuFun;
    typedef funct::Slave<funct::Product<funct::Parameter, ZPeak>::type> ZMuMuFunClone;
    typedef funct::Product<funct::Product<funct::Power<funct::Parameter, funct::Numerical<2> >::type, 
                                          funct::Power<funct::Parameter, funct::Numerical<2> >::type >::type, 
                           IsoefficiencytermSQ >::type  ZMuMuEfficiencyTerm;
    typedef funct::Product<ZMuMuEfficiencyTerm, ZMuMuFun>::type ZMuMuSig;


    typedef funct::Product<funct::Product<funct::Power<funct::Parameter, funct::Numerical<2> >::type, 
                                          funct::Power<funct::Parameter, funct::Numerical<2> >::type >::type, 
                          funct::Difference<funct::Numerical<1>, IsoefficiencytermSQ >::type>::type  ZMuMuNoIsoEfficiencyTerm;

    typedef funct::Product<ZMuMuNoIsoEfficiencyTerm, ZMuMuFunClone>::type ZMuMuNoIsoSig;
    
    typedef funct::Product<funct::Product<funct::Numerical<2>, 
                                          funct::Product<funct::Power<funct::Parameter, funct::Numerical<2> >::type, 
                                                         funct::Product<funct::Parameter, 
                                                                        funct::Difference<funct::Numerical<1>, funct::Parameter>::type 
                                                                       >::type 
                                                        >::type 
                           >::type,  IsoefficiencytermSQ >::type  ZMuTkEfficiencyTerm;



    typedef funct::Product<ZMuTkEfficiencyTerm, ZMuMuFunClone>::type ZMuTkSig;
    typedef funct::Product<funct::Parameter, 
                           funct::Product<funct::Exponential, funct::Polynomial<2> >::type >::type ZMuTkBkg;
    typedef funct::Product<funct::Constant,ZMuTkBkg>::type ZMuTkBkgScaled;//bgtrack rescaled
    typedef ZMuTkBkg ZMuMuNoIsoBkg;
    typedef ZMuTkBkgScaled  ZMuMuNoIsoBkgScaled ;//bgZmmNotIso rescaled
    typedef ZMuTkEfficiencyTerm ZMuSaEfficiencyTerm;
    typedef funct::Product<ZMuSaEfficiencyTerm, 
                           funct::Product<funct::Parameter, funct::Gaussian>::type>::type ZMuSaSig;
    typedef funct::Product<funct::Parameter, funct::Exponential>::type ZMuSaBkg;
 
    // typedef ZMuTkBkg ZMuMuNoIsoBkg;
    typedef funct::Product<funct::Constant, funct::Sum<ZMuMuNoIsoSig, ZMuMuNoIsoBkg>::type>::type ZMuMuNoIso;//3
    typedef funct::Product<funct::Constant, ZMuMuSig>::type ZMuMu;
    typedef funct::Product<funct::Constant, funct::Sum<ZMuTkSig, ZMuTkBkg>::type>::type ZMuTk;
    typedef funct::Product<funct::Constant, funct::Sum<ZMuSaSig, ZMuSaBkg>::type>::type ZMuSa;
    typedef fit::MultiHistoChiSquare<ZMuMu, ZMuTk, ZMuSa, ZMuMuNoIso> ChiSquared;

    double fMin, fMax;
    string ext;
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help,h", "produce help message")
      ("input-file,i", po::value< vector<string> >(), "input file")
      ("min,m", po::value<double>(&fMin)->default_value(60), "minimum value for fit range")
      ("max,M", po::value<double>(&fMax)->default_value(120), "maximum value for fit range")
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
    
    fit::RootMinuitCommands<ChiSquared> commands("csa08ZFit.txt");

    const int rebinMuMuNoIso = 2,rebinMuMu = 1, rebinMuTk = 2, rebinMuSa = 8;
    // assume that the bin size is 1 GeV!!!
    funct::Constant rebinMuMuNoIsoConst(rebinMuMuNoIso),rebinMuMuConst(rebinMuMu), rebinMuTkConst(rebinMuTk), rebinMuSaConst(rebinMuSa);

    if (vm.count("input-file")) {
      cout << "Input files are: " 
	   << vm["input-file"].as< vector<string> >() << "\n";
      vector<string> v_file = vm["input-file"].as< vector<string> >();
      for(vector<string>::const_iterator it = v_file.begin(); 
	  it != v_file.end(); ++it) {
	TFile * root_file = new TFile(it->c_str(),"read");

	TH1D * histoZMuMuNoIso = (TH1D*) root_file->Get("nonIsolatedZToMuMuPlots/zMass");
	histoZMuMuNoIso->Rebin(rebinMuMuNoIso);
	fix(histoZMuMuNoIso);

	TH1D * histoZMuMu = (TH1D*) root_file->Get("goodZToMuMuPlots/zMass");
	histoZMuMu->Rebin(rebinMuMu);
	fix(histoZMuMu);

	TH1D * histoZMuTk = (TH1D*) root_file->Get("goodZToMuMuOneTrackPlots/zMass");
	histoZMuTk->Rebin(rebinMuTk);
	fix(histoZMuTk);

	TH1D * histoZMuSa = (TH1D*) root_file->Get("goodZToMuMuOneStandAloneMuonPlots/zMass");
	histoZMuSa->Rebin(rebinMuSa);
	fix(histoZMuSa);

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
	const char * kYieldBkgZMuTk = "YieldBkgZMuTk"; 
	const char * kYieldBkgZMuSa = "YieldBkgZMuSa"; 
	const char * kYieldBkgZMuMuNotIso = "YieldBkgZMuMuNotIso"; 
	const char * kLambdaZMuMu = "LambdaZMuMu";
	const char * kMass = "Mass";
	const char * kGamma = "Gamma";
	const char * kPhotonFactorZMuMu = "PhotonFactorZMuMu";
	const char * kInterferenceFactorZMuMu = "InterferenceFactorZMuMu";
	const char * kMeanZMuMu = "MeanZMuMu";
	const char * kSigmaZMuMu = "SigmaZMuMu";
	const char * kAlpha = "Alpha";
	const char * kB0 = "B0"; 
	const char * kB1 = "B1"; 
	const char * kB2 = "B2"; 
	const char * kLambda = "Lambda";
	const char * kA0 = "A0"; 
	const char * kA1 = "A1"; 
	const char * kA2 = "A2"; 
	const char * kBeta = "Beta";
	const char * kSigmaZMuSa = "SigmaZMuSa";
	
	funct::Parameter lambdaZMuMu(kLambdaZMuMu, commands.par(kLambdaZMuMu));
	funct::Parameter mass(kMass, commands.par(kMass));
	funct::Parameter gamma(kGamma, commands.par(kGamma));
	funct::Parameter photonFactorZMuMu(kPhotonFactorZMuMu, commands.par(kPhotonFactorZMuMu)); 
	funct::Parameter interferenceFactorZMuMu(kInterferenceFactorZMuMu, commands.par(kInterferenceFactorZMuMu)); 
	funct::Parameter yieldZMuMu(kYieldZMuMu, commands.par(kYieldZMuMu));
	funct::Parameter efficiencyTk(kEfficiencyTk, commands.par(kEfficiencyTk)); 
	funct::Parameter efficiencySa(kEfficiencySa, commands.par(kEfficiencySa)); 
	funct::Parameter efficiencyIso(kEfficiencyIso, commands.par(kEfficiencyIso)); 
	funct::Parameter yieldBkgZMuTk(kYieldBkgZMuTk, commands.par(kYieldBkgZMuTk));
	funct::Parameter yieldBkgZMuSa(kYieldBkgZMuSa, commands.par(kYieldBkgZMuSa));
	funct::Parameter yieldBkgZMuMuNotIso(kYieldBkgZMuMuNotIso, commands.par(kYieldBkgZMuMuNotIso));
	funct::Parameter meanZMuMu(kMeanZMuMu, commands.par(kMeanZMuMu));
	funct::Parameter sigmaZMuMu(kSigmaZMuMu, commands.par(kSigmaZMuMu)); 
	funct::Parameter sigmaZMuSa(kSigmaZMuSa, commands.par(kSigmaZMuSa)); 
	funct::Parameter lambda(kLambda, commands.par(kLambda));
	funct::Parameter alpha(kAlpha, commands.par(kAlpha));
	funct::Parameter beta(kBeta, commands.par(kBeta));
	funct::Parameter b0(kB0, commands.par(kB0));
	funct::Parameter b1(kB1, commands.par(kB1));
	funct::Parameter b2(kB2, commands.par(kB2));
	funct::Parameter a0(kA0, commands.par(kA0));
	funct::Parameter a1(kA1, commands.par(kA1));
	funct::Parameter a2(kA2, commands.par(kA2));
	funct::Constant cFMin(fMin), cFMax(fMax);

	//IntegratorConv integratorConv(20);
	//IntegratorNorm integratorNorm(20);
	IntegratorConv integratorConv(1.e-4);
	IntegratorNorm integratorNorm(1.e-4);

	ZPeakNoNorm zPeakNN = funct::Exponential(lambdaZMuMu) * 
	  funct::conv(funct::ZLineShape(mass, gamma, photonFactorZMuMu, interferenceFactorZMuMu), 
		      funct::Gaussian(meanZMuMu, sigmaZMuMu), 
		      -3*sigmaZMuMu.value(), 3*sigmaZMuMu.value(), integratorConv);
	ZPeak zPeak = zPeakNN / ZPeakNormFactor(zPeakNN, cFMin, cFMax, integratorNorm);
	ZMuMuFun zMuMuFun = funct::master(yieldZMuMu * zPeak);
	ZMuMuFunClone zMuMuFunClone = funct::slave(zMuMuFun);
	IsoefficiencytermSQ efficiencyIsoSquare = (efficiencyIso ^ funct::Numerical<2>(2)); //efficienza Isolamento al quadrato
	ZMuMuEfficiencyTerm zMuMuEfficiencyTerm = ((efficiencyTk ^ funct::Numerical<2>(2)) * 
	  (efficiencySa ^ funct::Numerical<2>(2))) * efficiencyIsoSquare; 
	ZMuMuNoIsoEfficiencyTerm zMuMuNoIsoEfficiencyTerm = ((efficiencyTk ^ funct::Numerical<2>(2)) * 
	  (efficiencySa ^ funct::Numerical<2>(2))) * (funct::Numerical<1>(1) - efficiencyIsoSquare);
	ZMuMu zMuMu = rebinMuMuConst * (zMuMuEfficiencyTerm * zMuMuFun);

	ZMuTkBkg zMuTkBkg = yieldBkgZMuTk * (funct::Exponential(lambda) * funct::Polynomial<2>(a0, a1, a2));
	ZMuTkBkgScaled zMuTkBkgScaled = rebinMuTkConst * zMuTkBkg;
	ZMuTkEfficiencyTerm zMuTkEfficiencyTerm = funct::Numerical<2>(2) * 
	  ((efficiencyTk ^ funct::Numerical<2>(2)) * (efficiencySa * (funct::Numerical<1>(1) - efficiencySa))) * efficiencyIsoSquare;
	ZMuTk zMuTk = rebinMuTkConst*(zMuTkEfficiencyTerm * zMuMuFunClone + zMuTkBkg);

	ZMuMuNoIsoBkg zMuMuNoIsoBkg = yieldBkgZMuMuNotIso * (funct::Exponential(alpha) * funct::Polynomial<2>(b0, b1, b2));
	ZMuMuNoIsoBkgScaled  zMuMuNoIsoBkgScaled = rebinMuMuNoIsoConst * zMuMuNoIsoBkg;
	ZMuMuNoIso zMuMuNoIso = rebinMuMuNoIsoConst * ((zMuMuNoIsoEfficiencyTerm * zMuMuFunClone) +  zMuMuNoIsoBkg);

	ZMuSaEfficiencyTerm zMuSaEfficiencyTerm = funct::Numerical<2>(2) * 
	  ((efficiencySa ^ funct::Numerical<2>()) * (efficiencyTk * (funct::Numerical<1>() - efficiencyTk)))* efficiencyIsoSquare ;
	ZMuSa zMuSa = rebinMuSaConst *(zMuSaEfficiencyTerm * (yieldZMuMu * funct::Gaussian(mass, sigmaZMuSa)) 
				       + (yieldBkgZMuSa * funct::Exponential(beta)));

	ChiSquared chi2(zMuMu, histoZMuMu, 
			zMuTk, histoZMuTk, 
			zMuSa, histoZMuSa, 
			zMuMuNoIso,histoZMuMuNoIso,
			fMin, fMax);//WARNING attento all'ordine in cui hai definito il ch2
	cout << "N. deg. of freedom: " << chi2.numberOfBins() << endl;
	fit::RootMinuit<ChiSquared> minuit(chi2, true);
	commands.add(minuit, yieldZMuMu);
	commands.add(minuit, efficiencyTk);
	commands.add(minuit, efficiencySa);
	commands.add(minuit, efficiencyIso);
	commands.add(minuit, yieldBkgZMuTk);
	commands.add(minuit, yieldBkgZMuSa);
	commands.add(minuit, yieldBkgZMuMuNotIso);
	commands.add(minuit, lambdaZMuMu);
	commands.add(minuit, mass);
	commands.add(minuit, gamma);
	commands.add(minuit, photonFactorZMuMu);
	commands.add(minuit, interferenceFactorZMuMu);
	commands.add(minuit, meanZMuMu);
	commands.add(minuit, sigmaZMuMu);
	commands.add(minuit, sigmaZMuSa);
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
	const unsigned int nPar = 24;//WARNIG: this must be updated manually for now
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
	for(int i = 1; i <= histoZMuMuNoIso->GetNbinsX(); ++i)
	  s += histoZMuMuNoIso->GetBinContent(i);
	histoZMuMuNoIso->SetEntries(s);
	for(int i = 1; i <= histoZMuMu->GetNbinsX(); ++i)
	  s += histoZMuMu->GetBinContent(i);
	histoZMuMu->SetEntries(s);
	s = 0;
	for(int i = 1; i <= histoZMuTk->GetNbinsX(); ++i)
	  s += histoZMuTk->GetBinContent(i);
	histoZMuTk->SetEntries(s);
	s = 0;
	for(int i = 1; i <= histoZMuSa->GetNbinsX(); ++i)
	  s += histoZMuSa->GetBinContent(i);
	histoZMuSa->SetEntries(s);
	string ZMuMuPlot = "ZMuMuFit_" + plot_string;
	root::plot<ZMuMu>(ZMuMuPlot.c_str(), *histoZMuMu, zMuMu, fMin, fMax, 
			  efficiencyTk, efficiencySa, efficiencyIso,
			  yieldZMuMu, lambdaZMuMu, mass, gamma, photonFactorZMuMu, interferenceFactorZMuMu, 
			  meanZMuMu, sigmaZMuMu, 
			  kRed, 2, kDashed, 100, 
			  "Z -> #mu #mu mass", "#mu #mu invariant mass (GeV/c^{2})", 
			  "Events");
	
	string ZMuMuNoIsoPlot = "ZMuMuNoIsoFit_" + plot_string;
	root::plot<ZMuMuNoIso>(ZMuMuNoIsoPlot.c_str(), *histoZMuMuNoIso, zMuMuNoIso, fMin, fMax, 
			       efficiencyTk, efficiencySa, efficiencyIso, 
			       yieldZMuMu, lambdaZMuMu, mass, gamma, photonFactorZMuMu, interferenceFactorZMuMu, 
			       meanZMuMu, sigmaZMuMu, 
			       kRed, 2, kDashed, 100, 
			       "Z -> #mu #mu Not Iso mass", "#mu #mu invariant mass (GeV/c^{2})", 
			       "Events");	
	
	string ZMuTkPlot = "ZMuTkFit_" + plot_string;
	TF1 funZMuTk = root::tf1<ZMuTk>("ZMuTkFunction", zMuTk, fMin, fMax, 
					efficiencyTk, efficiencySa,efficiencyIso,
					yieldZMuMu, lambdaZMuMu, mass, gamma, photonFactorZMuMu, interferenceFactorZMuMu, 
					meanZMuMu, sigmaZMuMu, 
					yieldBkgZMuTk, lambda, a0, a1, a2);
	funZMuTk.SetLineColor(kRed);
	funZMuTk.SetLineWidth(2);
	funZMuTk.SetLineStyle(kDashed);
	funZMuTk.SetNpx(10000);
	TF1 funZMuTkBkg = root::tf1<ZMuTkBkgScaled>("ZMuTkBack", zMuTkBkgScaled, fMin, fMax, 
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
	funZMuTk.Draw("same");
	funZMuTkBkg.Draw("same");
	canvas->SaveAs(ZMuTkPlot.c_str());
	canvas->SetLogy();
	string logZMuTkPlot = "log_" + ZMuTkPlot;
	canvas->SaveAs(logZMuTkPlot.c_str());
	string ZMuSaPlot = "ZMuSaFit_" + plot_string;
	root::plot<ZMuSa>(ZMuSaPlot.c_str(), *histoZMuSa, zMuSa, fMin, fMax, 
			  efficiencySa, efficiencyTk, efficiencyIso,
			  yieldZMuMu, mass, sigmaZMuSa, yieldBkgZMuSa, 
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



